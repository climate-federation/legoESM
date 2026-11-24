#!/usr/bin/env python3
"""Fail-closed reader and 0-ULP production scorer for SI3 rung O4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants as canonical_constants
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.si3_orca2_state import (
    SI3AllPratherMoments,
    SI3ORCA2AdvectionGrid,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    si3_dyn_adv_pra_orca2,
    si3_lbc_lnk_orca2,
    si3_o4_step,
    si3_var_zapsmall,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import max_ulp_diff

MAGIC = b"O4PRA001"
HEADER_BYTES = 8 + 4 + 16 * 4 + 4 * 8
STEPS = (1, 2)

GRID = (
    "e1u", "e2v", "r1_e1u", "r1_e2v", "e2u", "e1v", "e1e2t",
    "r1_e1e2t", "tmask", "umask", "vmask", "u_ice", "v_ice", "sss_m", "sst_m",
)
STATE = (
    "ato_i", "h_i", "h_s", "h_ip", "h_il", "v_i", "v_s", "sv_i", "oa_i",
    "a_i", "a_ip", "v_ip", "v_il", "t_su", "e_s", "e_i", "szv_i", "sz_i", "t_i", "t_s",
    "at_i", "vt_i", "snwice_mass", "wfx_res", "hfx_res", "sfx_res", "diag_adv_mass",
    "diag_adv_salt", "diag_adv_heat", "diag_trp_vi", "diag_trp_vs", "diag_trp_ei",
    "diag_trp_es", "diag_trp_sv",
)
FAMILIES = ("ice", "sn", "a", "age", "sal", "ap", "vp", "vl", "c0", "e", "si")
COMPONENTS = ("sx", "sy", "sxx", "syy", "sxy")
MOMENTS = tuple(f"{family}.{component}" for family in FAMILIES for component in COMPONENTS)
KINDS = {1: GRID, 2: STATE, 3: MOMENTS, 4: (), 5: STATE, 6: MOMENTS,
         7: STATE, 8: STATE, 9: STATE}

# Used by the shared static read-set gate.  Entry kinds are 1--4; kinds 5--9
# are outputs and therefore cannot justify a production read.
PROJECTIONS = {(kind, name): name.replace(".", "_") for kind in KINDS for name in KINDS[kind]}
REPO_ROOT = Path(__file__).resolve().parents[4]

_JIT_ADVECT = jax.jit(
    si3_dyn_adv_pra_orca2,
    static_argnames=(
        "kt", "config", "_plant_invert_sweep_order", "_plant_perturb_hbig_cap",
        "_plant_omit_subcycle_lbc", "_plant_omit_donor_readjust",
        "_plant_corrupt_receiver_flux", "_plant_wrong_nfd_sign",
        "_plant_ignore_mppdelay",
    ),
)
_JIT_ZAPSMALL = jax.jit(si3_var_zapsmall, static_argnames=("nn_hls", "nn_icesal"))
_JIT_LBC = jax.jit(
    si3_lbc_lnk_orca2,
    static_argnames=("cd_nat", "psgn", "nn_hls", "ldfull", "_plant_wrong_nfd_sign"),
)
_JIT_STEP = jax.jit(
    si3_o4_step,
    static_argnames=(
        "kt", "rn_uice", "rn_vice", "config", "apply_halos",
        "_plant_invert_sweep_order", "_plant_perturb_hbig_cap",
        "_plant_omit_subcycle_lbc", "_plant_omit_donor_readjust",
        "_plant_corrupt_receiver_flux", "_plant_wrong_nfd_sign",
        "_plant_ignore_mppdelay", "_plant_stop_gradient",
    ),
)


def _refuse(message: str) -> ValueError:
    return ValueError(f"STATUS PLANT-FIRED: {message}")


def _shape(kind: int, name: str, meta: dict[str, int]) -> tuple[int, ...]:
    full = (meta["jpi"], meta["jpj"])
    interior = (meta["nie0"] - meta["nis0"] + 1, meta["nje0"] - meta["njs0"] + 1)
    if kind == 1 or name in {"ato_i", "at_i", "vt_i", "snwice_mass"}:
        return full
    if name in {"wfx_res", "hfx_res", "sfx_res", "diag_adv_mass", "diag_adv_salt", "diag_adv_heat",
                "diag_trp_vi", "diag_trp_vs", "diag_trp_ei", "diag_trp_es", "diag_trp_sv"}:
        return interior
    family = name.split(".", 1)[0]
    if family == "c0" or name in {"e_s", "t_s"}:
        return full + (meta["nlay_s"], meta["jpl"])
    if family in {"e", "si"} or name in {"e_i", "szv_i", "sz_i", "t_i"}:
        return full + (meta["nlay_i"], meta["jpl"])
    return full + (meta["jpl"],)


def read_record(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Read one O4 stream record and reject every undeclared byte or extent."""
    p = Path(path)
    raw = bytearray(p.read_bytes())
    if plant == "header" and raw:
        raw[0] ^= 1
    elif plant == "truncation" and raw:
        del raw[-1]
    elif plant == "step" and len(raw) >= 20:
        raw[16:20] = np.asarray([99], dtype="<i4").tobytes()
    if len(raw) < HEADER_BYTES or bytes(raw[:8]) != MAGIC:
        raise _refuse(f"bad/truncated O4 header: {p}")
    version = int(np.frombuffer(raw, dtype="<i4", count=1, offset=8)[0])
    if version != 1:
        raise _refuse(f"unsupported O4 version {version}")
    m = np.frombuffer(raw, dtype="<i4", count=16, offset=12).astype(np.int64)
    keys = ("kind", "kt", "rank", "jpi", "jpj", "jpl", "nlay_i", "nlay_s", "nn_hls",
            "nis0", "nie0", "njs0", "nje0", "nfields", "icycle", "sweep")
    meta = {key: int(value) for key, value in zip(keys, m, strict=True)}
    if meta["kind"] not in KINDS or meta["nfields"] != len(KINDS[meta["kind"]]):
        raise _refuse(f"unknown kind/count in {p}")
    if meta["kt"] not in STEPS or meta["sweep"] != (meta["kt"] - 1) % 2:
        raise _refuse(f"wrong step/sweep metadata in {p}")
    if min(meta[k] for k in ("jpi", "jpj", "jpl", "nlay_i", "nlay_s")) < 1:
        raise _refuse(f"non-positive O4 extent in {p}")
    if (meta["nie0"] - meta["nis0"] + 1 != meta["jpi"] - 2 * meta["nn_hls"] or
            meta["nje0"] - meta["njs0"] + 1 != meta["jpj"] - 2 * meta["nn_hls"]):
        raise _refuse(f"interior/full extent inconsistency in {p}")
    scalars = np.frombuffer(raw, dtype="<f8", count=4, offset=76).copy()
    if not np.all(np.isfinite(scalars)):
        raise _refuse(f"non-finite header scalar in {p}")
    offset = HEADER_BYTES
    fields: dict[str, np.ndarray] = {}
    for name in KINDS[meta["kind"]]:
        shape = _shape(meta["kind"], name, meta)
        count = int(np.prod(shape, dtype=np.int64))
        stop = offset + count * 8
        if stop > len(raw):
            raise _refuse(f"truncated field {name} in {p}")
        fields[name] = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).copy().reshape(shape, order="F")
        if not np.all(np.isfinite(fields[name])):
            raise _refuse(f"non-finite field {name} in {p}")
        offset = stop
    if offset != len(raw):
        raise _refuse(f"undeclared trailing bytes in {p}")
    return {"meta": meta, "scalars": scalars, "fields": fields, "sha256": hashlib.sha256(raw).hexdigest()}


def load_deck(root: str | Path, plant: str | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    root = Path(root)
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for step in STEPS:
        for kind in KINDS:
            path = root / f"o4_kind{kind:02d}_step{step:04d}_rank0000.bin"
            if not path.is_file():
                raise _refuse(f"missing declared O4 record {path.name}")
            records[(step, kind)] = read_record(path, plant if (step, kind) == (1, 1) else None)
    extra = set(root.glob("o4_kind*_step*_rank*.bin")) - {
        root / f"o4_kind{kind:02d}_step{step:04d}_rank0000.bin" for step in STEPS for kind in KINDS
    }
    if extra:
        raise _refuse(f"undeclared O4 records: {sorted(p.name for p in extra)}")
    base = records[(1, 1)]["meta"]
    invariant = ("rank", "jpi", "jpj", "jpl", "nlay_i", "nlay_s", "nn_hls", "nis0", "nie0", "njs0", "nje0")
    for rec in records.values():
        if any(rec["meta"][key] != base[key] for key in invariant):
            raise _refuse("cross-record O4 metadata mismatch")
    return records


def _moments(fields: dict[str, np.ndarray]) -> SI3AllPratherMoments:
    return SI3AllPratherMoments(*(
        SI3PratherMoments(*(jnp.asarray(fields[f"{family}.{c}"]) for c in COMPONENTS))
        for family in FAMILIES
    ))


def _flatten_moments(value: SI3AllPratherMoments) -> dict[str, np.ndarray]:
    return {f"{family}.{component}": np.asarray(getattr(getattr(value, family), component))
            for family in FAMILIES for component in COMPONENTS}


def _pad_interior(a: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    h = meta["nn_hls"]
    out = np.zeros((meta["jpi"], meta["jpj"]), dtype=np.float64)
    out[h:meta["jpi"] - h, h:meta["jpj"] - h] = a
    return out


def _crop_interior(a: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    h = meta["nn_hls"]
    return a[h:meta["jpi"] - h, h:meta["jpj"] - h]


RESIDUALS = ("wfx_res", "hfx_res", "sfx_res", "diag_adv_mass", "diag_adv_salt", "diag_adv_heat")
TRANSPORT_DIAGNOSTICS = ("diag_trp_vi", "diag_trp_vs", "diag_trp_ei", "diag_trp_es", "diag_trp_sv")


def _build_entry_state(entry: dict[str, np.ndarray], moments: dict[str, np.ndarray],
                       grid_f: dict[str, np.ndarray], cfg: SI3ORCA2StateConfig,
                       delayed_cfl: float) -> SI3State:
    base = SI3State.zeros(entry["v_i"].shape[0], entry["v_i"].shape[1], config=cfg)
    eqv = base.equivalent._replace(
        h_i=jnp.asarray(entry["h_i"]), h_s=jnp.asarray(entry["h_s"]),
        h_ip=jnp.asarray(entry["h_ip"]), h_il=jnp.asarray(entry["h_il"]),
        sz_i=jnp.asarray(entry["sz_i"]), t_i=jnp.asarray(entry["t_i"]),
        t_s=jnp.asarray(entry["t_s"]), a_i=jnp.asarray(entry["a_i"]),
        v_s=jnp.asarray(entry["v_s"]),
    )
    agg = base.aggregated._replace(
        ato_i=jnp.asarray(entry["ato_i"]), at_i=jnp.asarray(entry["at_i"]),
        vt_i=jnp.asarray(entry["vt_i"]),
    )
    return base._replace(
        v_i=jnp.asarray(entry["v_i"]), v_s=jnp.asarray(entry["v_s"]),
        sv_i=jnp.asarray(entry["sv_i"]), a_i=jnp.asarray(entry["a_i"]),
        t_su=jnp.asarray(entry["t_su"]), u_ice=jnp.asarray(grid_f["u_ice"]),
        v_ice=jnp.asarray(grid_f["v_ice"]), oa_i=jnp.asarray(entry["oa_i"]),
        a_ip=jnp.asarray(entry["a_ip"]), v_ip=jnp.asarray(entry["v_ip"]),
        v_il=jnp.asarray(entry["v_il"]), e_s=jnp.asarray(entry["e_s"]),
        e_i=jnp.asarray(entry["e_i"]), szv_i=jnp.asarray(entry["szv_i"]),
        snwice_mass=jnp.asarray(entry["snwice_mass"]), prather=_moments(moments),
        equivalent=eqv, aggregated=agg, cflice_delayed=jnp.asarray(delayed_cfl),
    )


def _state_and_diagnostics(state: SI3State, diagnostics: Any) -> dict[str, np.ndarray]:
    out = {
        "ato_i": np.asarray(state.aggregated.ato_i), "h_i": np.asarray(state.equivalent.h_i),
        "h_s": np.asarray(state.equivalent.h_s), "h_ip": np.asarray(state.equivalent.h_ip),
        "h_il": np.asarray(state.equivalent.h_il), "v_i": np.asarray(state.v_i),
        "v_s": np.asarray(state.v_s), "sv_i": np.asarray(state.sv_i), "oa_i": np.asarray(state.oa_i),
        "a_i": np.asarray(state.a_i), "a_ip": np.asarray(state.a_ip), "v_ip": np.asarray(state.v_ip),
        "v_il": np.asarray(state.v_il), "t_su": np.asarray(state.t_su), "e_s": np.asarray(state.e_s),
        "e_i": np.asarray(state.e_i), "szv_i": np.asarray(state.szv_i),
        "sz_i": np.asarray(state.equivalent.sz_i), "t_i": np.asarray(state.equivalent.t_i),
        "t_s": np.asarray(state.equivalent.t_s), "at_i": np.asarray(state.aggregated.at_i),
        "vt_i": np.asarray(state.aggregated.vt_i), "snwice_mass": np.asarray(state.snwice_mass),
    }
    out.update({name: np.asarray(getattr(diagnostics, name)) for name in RESIDUALS + TRANSPORT_DIAGNOSTICS})
    return out


def _array_metrics(actual: np.ndarray, expected: np.ndarray,
                   mask: np.ndarray) -> dict[str, Any]:
    """Return strict float64 bit metrics over one broadcast wet mask."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise _refuse(f"score shape mismatch: {actual.shape} != {expected.shape}")
    selected = np.asarray(mask, dtype=bool)
    while selected.ndim < actual.ndim:
        selected = selected[..., None]
    selected = np.broadcast_to(selected, actual.shape)
    cells = int(np.count_nonzero(selected))
    if cells == 0:
        raise _refuse("score region has zero wet scalar cells")
    unequal_mask = selected & (actual.view(np.uint64) != expected.view(np.uint64))
    unequal = int(np.count_nonzero(unequal_mask))
    first = None
    if unequal:
        first = [int(value) for value in np.argwhere(unequal_mask)[0]]
    return {
        "cells": cells,
        "unequal": unequal,
        "max_ulp": max_ulp_diff(actual, expected, selected, strict_signed_zero=True),
        "first_index": first,
    }


def _new_row(step: int, record_kind: int, stage: str, region: str) -> dict[str, Any]:
    return {
        "step": step,
        "record_kind": record_kind,
        "stage": stage,
        "region": region,
        "cells": 0,
        "unequal": 0,
        "max_ulp": 0,
        "first_differing_cell": None,
    }


def _add_metrics(row: dict[str, Any], field: str, metrics: dict[str, Any]) -> None:
    row["cells"] += metrics["cells"]
    row["unequal"] += metrics["unequal"]
    row["max_ulp"] = max(row["max_ulp"], metrics["max_ulp"])
    if row["first_differing_cell"] is None and metrics["first_index"] is not None:
        row["first_differing_cell"] = {
            "field": field,
            "index": metrics["first_index"],
        }


def _compare_stage(step: int, record_kind: int, label: str,
                   got: dict[str, np.ndarray], expected: dict[str, np.ndarray],
                   meta: dict[str, int], wet: np.ndarray, seam: np.ndarray,
                   fold: np.ndarray) -> list[dict[str, Any]]:
    rows = {region: _new_row(step, record_kind, label, region)
            for region in ("full", "seam", "fold")}
    for name in STATE:
        actual = got[name]
        if name in RESIDUALS + TRANSPORT_DIAGNOSTICS:
            h = meta["nn_hls"]
            field_wet = wet[h:-h, h:-h]
            masks = {
                "full": field_wet,
                "seam": seam[h:-h, h:-h] & field_wet,
                "fold": fold[h:-h, h:-h] & field_wet,
            }
        else:
            masks = {"full": wet, "seam": seam & wet, "fold": fold & wet}
        for region, mask in masks.items():
            _add_metrics(rows[region], name, _array_metrics(actual, expected[name], mask))
    return list(rows.values())


def _compare_moments(step: int, got: dict[str, np.ndarray],
                     expected: dict[str, np.ndarray], wet: np.ndarray,
                     seam: np.ndarray, fold: np.ndarray) -> list[dict[str, Any]]:
    rows = {region: _new_row(step, 6, "advection_moments", region)
            for region in ("full", "seam", "fold")}
    masks = {"full": wet, "seam": seam & wet, "fold": fold & wet}
    for name in MOMENTS:
        for region, mask in masks.items():
            _add_metrics(rows[region], name, _array_metrics(got[name], expected[name], mask))
    return list(rows.values())


def _control_row(step: int, out: tuple[Any, ...], cfl: dict[str, Any]) -> dict[str, Any]:
    row = _new_row(step, 4, "controls", "full")
    controls = (
        ("delayed_cfl", np.asarray(out[19]), np.asarray(cfl["scalars"][3])),
        ("current_cfl", np.asarray(out[20]), np.asarray(cfl["scalars"][2])),
        ("icycle", np.asarray(out[21], dtype=np.float64),
         np.asarray(cfl["meta"]["icycle"], dtype=np.float64)),
        ("sweep", np.asarray((step - 1) % 2, dtype=np.float64),
         np.asarray(cfl["meta"]["sweep"], dtype=np.float64)),
    )
    for name, actual, expected in controls:
        _add_metrics(row, name, _array_metrics(actual, expected, np.asarray(True)))
    return row


def score_step(records: dict[tuple[int, int], dict[str, Any]], step: int,
               plant: str | None = None, mode: str = "eager") -> dict[str, Any]:
    """Replay every registered O4 boundary and demand strict signed-zero 0 ULP."""
    grid_f = records[(step, 1)]["fields"]
    entry = records[(step, 2)]["fields"]
    moment_entry = records[(step, 3)]["fields"]
    cfl = records[(step, 4)]
    meta = records[(step, 1)]["meta"]
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(
        jpl=meta["jpl"], nlay_i=meta["nlay_i"], nlay_s=meta["nlay_s"],
        nn_hls=meta["nn_hls"], nn_icesal=4, ln_icedyn=True, ln_icethd=False,
        ln_dynadv_pra=True, rDt_ice=float(cfl["scalars"][0]),
    )
    grid = SI3ORCA2AdvectionGrid(*(jnp.asarray(grid_f[name]) for name in GRID[:11]))
    kwargs = {name: jnp.asarray(_pad_interior(entry[name], meta)) for name in RESIDUALS}
    hooks: dict[str, Any] = {}
    if plant == "invert-sweep":
        hooks["_plant_invert_sweep_order"] = True
    elif plant == "wrong-fold-sign":
        hooks["_plant_wrong_nfd_sign"] = "sx"
    elif plant == "omit-subcycle-lbc":
        hooks["_plant_omit_subcycle_lbc"] = True
    advect = _JIT_ADVECT if mode == "jit" else si3_dyn_adv_pra_orca2
    out = advect(
        kt=step, u_ice=jnp.asarray(grid_f["u_ice"]), v_ice=jnp.asarray(grid_f["v_ice"]),
        h_i=jnp.asarray(entry["h_i"]), h_s=jnp.asarray(entry["h_s"]), h_ip=jnp.asarray(entry["h_ip"]),
        ato_i=jnp.asarray(entry["ato_i"]), v_i=jnp.asarray(entry["v_i"]), v_s=jnp.asarray(entry["v_s"]),
        sv_i=jnp.asarray(entry["sv_i"]), oa_i=jnp.asarray(entry["oa_i"]), a_i=jnp.asarray(entry["a_i"]),
        a_ip=jnp.asarray(entry["a_ip"]), v_ip=jnp.asarray(entry["v_ip"]), v_il=jnp.asarray(entry["v_il"]),
        e_s=jnp.asarray(entry["e_s"]), e_i=jnp.asarray(entry["e_i"]), szv_i=jnp.asarray(entry["szv_i"]),
        prather=_moments(moment_entry), grid=grid, config=cfg, hi_max_top=float(cfl["scalars"][1]),
        cflice_delayed=jnp.asarray(cfl["scalars"][3]), **kwargs, **hooks,
    )
    actual_adv = {name: np.asarray(entry[name]) for name in STATE}
    for name, value in zip(("ato_i", "v_i", "v_s", "sv_i", "oa_i", "a_i", "a_ip", "v_ip", "v_il",
                            "e_s", "e_i", "szv_i"), out[:12], strict=True):
        actual_adv[name] = np.asarray(value)
    actual_adv.update({name: _crop_interior(np.asarray(value), meta)
                       for name, value in zip(RESIDUALS, out[13:19], strict=True)})
    actual_moments = _flatten_moments(out[12])

    zapsmall = _JIT_ZAPSMALL if mode == "jit" else si3_var_zapsmall
    zap = zapsmall(
        v_i=out[1], v_s=out[2], sv_i=out[3], oa_i=out[4], a_i=out[5], a_ip=out[6],
        v_ip=out[7], v_il=out[8], t_su=jnp.asarray(entry["t_su"]),
        h_i=jnp.asarray(entry["h_i"]), h_s=jnp.asarray(entry["h_s"]),
        h_ip=jnp.asarray(entry["h_ip"]), h_il=jnp.asarray(entry["h_il"]),
        e_s=out[9], e_i=out[10], szv_i=out[11], sz_i=jnp.asarray(entry["sz_i"]),
        t_i=jnp.asarray(entry["t_i"]), t_s=jnp.asarray(entry["t_s"]),
        at_i=jnp.asarray(entry["at_i"]), vt_i=jnp.asarray(entry["vt_i"]), ato_i=out[0],
        wfx_res=out[13], hfx_res=out[14], sfx_res=out[15], rDt_ice=float(cfl["scalars"][0]),
        sss_m=jnp.asarray(grid_f["sss_m"]), sst_m=jnp.asarray(grid_f["sst_m"]),
        nn_hls=meta["nn_hls"], nn_icesal=4, rn_simin=cfg.rn_simin, rn_sinew=cfg.rn_sinew,
        rt0=cfg.ice_constants.T0, rhoi=cfg.ice_constants.rho_ice,
        rhos=cfg.ice_constants.rho_snow, rhow=canonical_constants.rho_water,
    )
    actual_zap = dict(actual_adv)
    for name, value in zip(("v_i", "v_s", "sv_i", "oa_i", "a_i", "a_ip", "v_ip", "v_il", "t_su",
                            "h_i", "h_s", "h_ip", "h_il", "e_s", "e_i", "szv_i", "sz_i", "t_i", "t_s",
                            "at_i", "vt_i", "ato_i", "wfx_res", "hfx_res", "sfx_res"), zap, strict=True):
        actual_zap[name] = _crop_interior(np.asarray(value), meta) if name in RESIDUALS else np.asarray(value)

    entry_state = _build_entry_state(entry, moment_entry, grid_f, cfg, float(cfl["scalars"][3]))
    production_step = _JIT_STEP if mode == "jit" else si3_o4_step
    final_state, final_diag = production_step(
        entry_state, grid, kt=step, sss_m=jnp.asarray(grid_f["sss_m"]),
        sst_m=jnp.asarray(grid_f["sst_m"]), rn_uice=0.5, rn_vice=0.5,
        config=cfg, apply_halos=False, **hooks,
    )
    for name in TRANSPORT_DIAGNOSTICS:
        actual_zap[name] = _crop_interior(np.asarray(getattr(final_diag, name)), meta)
    actual_lbc = dict(actual_zap)
    for name in ("a_i", "v_i", "v_s", "sv_i", "oa_i", "t_su", "e_i", "e_s", "szv_i"):
        lbc = _JIT_LBC if mode == "jit" else si3_lbc_lnk_orca2
        actual_lbc[name] = np.asarray(lbc(
            jnp.asarray(actual_zap[name]), "T", 1.0, nn_hls=meta["nn_hls"], ldfull=True))
    actual_final = _state_and_diagnostics(final_state, final_diag)
    for name in RESIDUALS + TRANSPORT_DIAGNOSTICS:
        actual_final[name] = _crop_interior(actual_final[name], meta)

    seam = np.zeros((meta["jpi"], meta["jpj"]), dtype=bool)
    h = meta["nn_hls"]
    seam[:2 * h, :] = True; seam[-2 * h:, :] = True
    fold = np.zeros_like(seam); fold[:, -3 * h:] = True
    wet = np.asarray(grid_f["tmask"]) > 0
    if (not np.any(seam & wet) or not np.any(fold & wet) or
            not np.any(seam[h:-h, h:-h] & wet[h:-h, h:-h]) or
            not np.any(fold[h:-h, h:-h] & wet[h:-h, h:-h])):
        raise _refuse("empty wet seam/fold score region")
    expected_by_stage = {"adv": dict(records[(step, 5)]["fields"]),
                         "zapsmall": dict(records[(step, 7)]["fields"]),
                         "lbc": dict(records[(step, 8)]["fields"]),
                         "step": dict(records[(step, 9)]["fields"])}
    plant_stage = {"one-bit": "adv", "one-bit-zapsmall": "zapsmall",
                   "one-bit-lbc": "lbc", "one-bit-step": "step"}.get(plant)
    if plant_stage:
        target = expected_by_stage[plant_stage]["v_i"].copy()
        wet_indices = np.argwhere(wet)
        i, j = wet_indices[len(wet_indices) // 2]
        target[i, j, 0] = np.nextafter(target[i, j, 0], np.inf)
        expected_by_stage[plant_stage]["v_i"] = target
    rows = [_control_row(step, out, cfl)]
    rows.extend(_compare_stage(step, 5, "advection", actual_adv,
                               expected_by_stage["adv"], meta, wet, seam, fold))
    rows.extend(_compare_moments(step, actual_moments,
                                 records[(step, 6)]["fields"], wet, seam, fold))
    for record_kind, label, actual in (
        (7, "zapsmall", actual_zap),
        (8, "lbc", actual_lbc),
        (9, "step", actual_final),
    ):
        rows.extend(_compare_stage(step, record_kind, label, actual,
                                   expected_by_stage[label], meta, wet, seam, fold))
    unequal = sum(row["unequal"] for row in rows)
    max_ulp = max(row["max_ulp"] for row in rows)
    if plant is not None and unequal:
        first = next(row for row in rows if row["unequal"])
        raise _refuse(
            f"O4 step {step} plant is not 0 ULP: kind={first['record_kind']} "
            f"stage={first['stage']} first={first['first_differing_cell']}"
        )
    return {
        "step": step,
        "sweep": "x-y" if step == 1 else "y-x",
        "status": "PASS_O4_EXACT" if unequal == 0 else "O4_DEBT",
        "cells": sum(row["cells"] for row in rows),
        "unequal": unequal,
        "max_ulp": max_ulp,
        "rows": rows,
    }


def score_deck(root: str | Path, plant: str | None = None,
               mode: str = "eager") -> dict[str, Any]:
    if mode not in {"eager", "jit"}:
        raise ValueError(f"unknown O4 execution mode {mode!r}")
    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    records = load_deck(root, plant if plant in {"header", "truncation", "step"} else None)
    if mode == "eager":
        with jax.disable_jit():
            scores = [score_step(records, step, plant, mode) for step in STEPS]
    else:
        scores = [score_step(records, step, plant, mode) for step in STEPS]
    status = "PASS_O4_EXACT" if all(score["status"] == "PASS_O4_EXACT" for score in scores) else "O4_DEBT"
    return {
        "status": status,
        "criterion": "0 ULP on every NEMO-written wet cell; seam and fold reported separately",
        "execution_mode": mode,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dtype": "float64",
        "scores": scores,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, raw: str, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise _refuse(f"{label} path is not relative: {raw}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise _refuse(f"{label} path escapes evidence: {raw}") from exc
    return resolved


def admit_evidence(root: str | Path, expect_commit: str | None = None,
                   plant: str | None = None) -> dict[str, Any]:
    """Admit a completed reference/record pair against its pre-run manifest."""
    root = Path(root)
    manifest_path = root / "manifest/run_manifest.json"
    if not manifest_path.is_file():
        raise _refuse("missing pre-run manifest")
    manifest = json.loads(manifest_path.read_text())
    required = {"producer_commit", "binary_sha256", "binary_paths", "source_sha256",
                "deck_sha256", "input_sha256", "planned_records", "steps", "ranks"}
    if required - set(manifest):
        raise _refuse(f"manifest missing keys {sorted(required-set(manifest))}")
    producer = str(manifest["producer_commit"])
    if plant == "producer":
        producer = "0" * 40
    if expect_commit is not None and producer != expect_commit:
        raise _refuse(f"producer commit mismatch: {producer} != {expect_commit}")
    try:
        subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{producer}^{{commit}}"],
                       check=True, capture_output=True)
    except subprocess.SubprocessError as exc:
        raise _refuse(f"producer commit is unavailable: {producer}") from exc
    for label, raw in manifest["binary_paths"].items():
        path = _relative(root, raw, f"binary {label}")
        if not path.is_file():
            raise _refuse(f"missing retained {label} binary")
        want = manifest["binary_sha256"].get(label)
        if plant == "binary" and label == sorted(manifest["binary_paths"])[0]:
            want = "0" * 64
        if _sha(path) != want:
            raise _refuse(f"retained {label} binary digest mismatch")
    fixture = bool(manifest.get("fixture", False)) and os.environ.get("O4_DRIVER_FIXTURE_ACTIVE") == "1"
    for raw, want in manifest["source_sha256"].items():
        if Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise _refuse(f"source path is not repository-relative: {raw}")
        if fixture:
            got = _sha(REPO_ROOT / raw)
        else:
            try:
                blob = subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{producer}:{raw}"])
            except subprocess.SubprocessError as exc:
                raise _refuse(f"source is not pinned by producer commit: {raw}") from exc
            got = hashlib.sha256(blob).hexdigest()
        if got != want:
            raise _refuse(f"producer source digest mismatch: {raw}")
    for section in ("deck_sha256", "input_sha256"):
        for raw, want in manifest[section].items():
            path = _relative(root, raw, section)
            if not path.is_file() or _sha(path) != want:
                raise _refuse(f"{section} digest mismatch: {raw}")
    if plant == "missing-complete" or not (root / "RUN_COMPLETE").is_file():
        raise _refuse("missing acquisition RUN_COMPLETE")
    for lane in ("reference_2step", "record_2step"):
        run = root / lane
        for name in ("ocean.output", "run.user.stdout.log", "run.user.stderr.log",
                     "run.user.time.log", "RUN_COMPLETE"):
            if not (run / name).is_file():
                raise _refuse(f"missing {lane}/{name}")
        joined = (run / "ocean.output").read_text(errors="replace") + "\n" + \
                 (run / "run.user.stdout.log").read_text(errors="replace") + "\n" + \
                 (run / "run.user.stderr.log").read_text(errors="replace")
        if plant == "error-log" and lane == "record_2step":
            joined += "\nE R R O R planted"
        if __import__("re").search(r"E\s+R\s+R\s+O\s+R|ABORT|STOP\s+[1-9]", joined, __import__("re").I):
            raise _refuse(f"fatal NEMO token in {lane}")
        if "STOP 0" not in joined or "MPIRUN_RC=0" not in (run / "run.user.time.log").read_text():
            raise _refuse(f"{lane} lacks STOP 0 / successful launcher status")
    ref_restarts = sorted((root / "reference_2step").glob("ORCA2_SAS_00000002_restart*.nc"))
    rec_restarts = sorted((root / "record_2step").glob("ORCA2_SAS_00000002_restart*.nc"))
    if not ref_restarts or [p.name for p in ref_restarts] != [p.name for p in rec_restarts]:
        raise _refuse("reference/record restart sets differ")
    for ref, rec in zip(ref_restarts, rec_restarts, strict=True):
        if plant == "restart" and rec.name.endswith("restart_ice.nc"):
            raise _refuse("observer changed planted ice restart")
        if ref.read_bytes() != rec.read_bytes():
            raise _refuse(f"observer changed restart bytes: {ref.name}")
    records = root / "record_2step/o4_prather_records"
    expected = set(manifest["planned_records"])
    actual = {p.name for p in records.glob("*.bin")}
    if plant == "missing-frame" and actual:
        actual.remove(sorted(actual)[0])
    if actual != expected:
        raise _refuse(f"record set mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    score_plant = "one-bit" if plant == "wet-1ulp" else None
    score = score_deck(records, score_plant, mode="eager")
    if score["status"] != "PASS_O4_EXACT":
        raise _refuse("admitted O4 evidence is not exact")
    checksums = root / "manifest/artifacts.sha256"
    if not checksums.is_file():
        raise _refuse("missing artifact checksum list")
    for line in checksums.read_text().splitlines():
        want, raw = line.split(None, 1)
        path = _relative(root, raw.strip(), "checksum")
        if not path.is_file() or _sha(path) != ("0" * 64 if plant == "checksum" else want):
            raise _refuse(f"artifact checksum mismatch: {raw.strip()}")
    return {"status": "PASS_O4_ADMISSION", "producer_commit": producer,
            "records": len(actual), "restarts": [p.name for p in ref_restarts], "score": score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="?")
    parser.add_argument("--admit-evidence", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--mode", choices=("eager", "jit"), default="eager")
    parser.add_argument("--plant", choices=("header", "truncation", "step", "one-bit", "one-bit-zapsmall", "one-bit-lbc", "one-bit-step",
                                             "invert-sweep", "wrong-fold-sign", "omit-subcycle-lbc",
                                             "producer", "binary", "checksum", "missing-complete", "error-log", "restart", "missing-frame", "wet-1ulp"))
    args = parser.parse_args()
    if args.admit_evidence:
        result = admit_evidence(args.admit_evidence, args.expect_commit, args.plant)
    elif args.records:
        result = score_deck(args.records, args.plant, args.mode)
    else:
        parser.error("records or --admit-evidence is required")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "O4_DEBT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
