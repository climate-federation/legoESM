#!/usr/bin/env python3
"""Fail-closed reader and exact-input admission gate for SI3 rung O7."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.dynamics import SI3ORCA2FoldProgram
from legoesm.ice.si3_surface import (
    SI3SurfaceConfig,
    orca2_flx_other_neighbours,
    si3_blk_ice_1,
    si3_blk_ice_2,
    si3_ice_alb,
    si3_ice_flx_other,
    si3_ice_sbc_tau_lbc,
    si3_surface_step,
)

from legoesm import constants
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    GateError as SameProgramError,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    same_program_observers,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import (
    first_wet_ij,
    max_normalized_diff,
    max_ulp_diff,
    verify_dry_signed_zero_fixture,
    verify_ocean_output_log,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MAGIC = b"O7SI3R01"
VERSION = 1
N_META = 20
N_SCALARS = 7
HEADER_BYTES = 8 + 4 + 4 * N_META + 8 * N_SCALARS
STEPS = (1, 2)
OBSERVER_UNITS = ("icestp", "sbcblk", "icealb", "icesbc")
WRITER_UNIT = "o7_si3_writer"
MARKERS = {unit: ("! O7_OBSERVER_BEGIN", "! O7_OBSERVER_END") for unit in OBSERVER_UNITS}

# Extent codes: I = NEMO-filled interior, F = declared full halo; rank follows.
KINDS: dict[int, tuple[tuple[str, str], ...]] = {
    1: tuple(
        (name, "I2")
        for name in ("wndi", "wndj", "theta_air_zt", "q_air_zt", "slp", "tm_su", "rhoa", "smask0")
    ),
    2: tuple(
        (name, "I2")
        for name in (
            "wndm_ice",
            "zsipt",
            "Cd_ice",
            "Ch_ice",
            "Ce_ice",
            "theta_zu_i",
            "q_zu_i",
            "utau_ice",
            "vtau_ice",
        )
    ),
    3: (("utau_ice", "F2"), ("vtau_ice", "F2")),
    4: tuple((name, "I3") for name in ("t_su", "h_i", "h_s", "a_ip_eff", "h_ip", "h_il"))
    + (("cloud_fra", "I2"), ("smask0", "I2")),
    5: (("alb_ice", "I3"),),
    6: tuple((name, "I3") for name in ("t_su", "h_s", "h_i", "alb_ice", "a_i_b"))
    + tuple(
        (name, "I2")
        for name in (
            "theta_air_zt",
            "q_air_zt",
            "slp",
            "qlw",
            "precip",
            "snow",
            "qsr",
            "rhoa",
            "wndm_ice",
            "Ch_ice",
            "Ce_ice",
            "theta_zu_i",
            "q_zu_i",
            "sst_m",
            "emp",
            "at_i_b",
            "qns_oce",
            "qsr_oce",
            "cloud_fra",
            "smask0",
        )
    ),
    7: tuple(
        (name, "I3")
        for name in (
            "qsr_ice",
            "qns_ice",
            "dqns_ice",
            "qla_ice",
            "dqla_ice",
            "evap_ice",
            "devap_ice",
            "qevap_ice",
            "qtr_ice_top",
            "zsq",
            "zsipt",
            "z_qlw",
            "z_dqlw",
            "z_qsb",
            "z_dqsb",
        )
    )
    + tuple(
        (name, "I2")
        for name in (
            "tprecip",
            "sprecip",
            "emp_oce",
            "emp_ice",
            "emp_tot",
            "qemp_oce",
            "qemp_ice",
            "qns_tot",
            "qsr_tot",
            "qprec_ice",
            "zcptrain",
            "zcptsnw",
            "zcptn",
            "zevap",
            "zsnw",
        )
    ),
    8: tuple((name, "F2") for name in ("u_ice", "v_ice", "ssu_m", "ssv_m", "utau", "vtau"))
    + tuple(
        (name, "I2")
        for name in (
            "smask0",
            "at_i",
            "at_i_b",
            "vt_i",
            "qsr_oce",
            "qns_oce",
            "qemp_oce",
            "frq_m",
            "e3t_m",
            "sst_m",
            "t_bo",
            "rn_amax_2d",
        )
    ),
    9: tuple((name, "I2") for name in ("qsb_ice_bot", "fhld", "qlead", "frq_m")),
}


def _refuse(message: str) -> ValueError:
    return ValueError(f"STATUS PLANT-FIRED: {message}")


def _shape(code: str, meta: dict[str, int]) -> tuple[int, ...]:
    ni, nj = (meta["ni"], meta["nj"]) if code[0] == "I" else (meta["jpi"], meta["jpj"])
    return (ni, nj, meta["jpl"]) if code[1] == "3" else (ni, nj)


def read_o7_record(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Read one writer record using only its declared extents and schema."""
    p = Path(path)
    raw = p.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    if len(raw) < HEADER_BYTES or raw[:8] != MAGIC:
        raise _refuse(f"bad/truncated magic in {p}")
    version = int(np.frombuffer(raw, dtype="<i4", count=1, offset=8)[0])
    if plant == "header":
        version += 1
    if version != VERSION:
        raise _refuse(f"unsupported record version {version}")
    values = np.frombuffer(raw, dtype="<i4", count=N_META, offset=12)
    keys = (
        "kind",
        "kt",
        "kmm",
        "rank",
        "jpi",
        "jpj",
        "jpl",
        "nn_hls",
        "ntsi",
        "ntei",
        "ntsj",
        "ntej",
        "ni",
        "nj",
        "nfields",
        "endian",
        "nit000",
        "jpni",
        "jpnj",
        "reserved",
    )
    meta = dict(zip(keys, map(int, values), strict=True))
    if plant == "extent":
        meta["ni"] += 1
    kind = meta["kind"]
    if kind not in KINDS or meta["nfields"] != len(KINDS[kind]):
        raise _refuse(f"unknown kind/field count {kind}/{meta['nfields']}")
    if not (
        meta["jpl"] == 5
        and meta["nn_hls"] == 2
        and meta["rank"] == 0
        and meta["jpni"] == 1
        and meta["jpnj"] == 1
        and meta["endian"] == 1
    ):
        raise _refuse(f"wrong production decomposition/extents: {meta}")
    if (
        meta["ni"] != meta["ntei"] - meta["ntsi"] + 1
        or meta["nj"] != meta["ntej"] - meta["ntsj"] + 1
    ):
        raise _refuse("interior extents disagree with filled bounds")
    if (meta["ntsi"], meta["ntei"], meta["ntsj"], meta["ntej"]) != (
        meta["nn_hls"] + 1,
        meta["jpi"] - meta["nn_hls"],
        meta["nn_hls"] + 1,
        meta["jpj"] - meta["nn_hls"],
    ):
        raise _refuse("filled range is not the one-rank NEMO interior")
    match = re.search(r"o7_kind(\d+)_step(\d+)_rank(\d+)\.bin$", p.name)
    if not match or tuple(map(int, match.groups())) != (kind, meta["kt"], meta["rank"]):
        raise _refuse(f"filename/header mismatch in {p.name}")
    scalars = np.frombuffer(raw, dtype="<f8", count=N_SCALARS, offset=12 + 4 * N_META)
    if not np.isfinite(scalars).all():
        raise _refuse(f"non-finite run scalar in {p.name}")
    scalar_names = ("rDt_ice", "r1_Dt_ice", "rho0", "rcp", "rn_Cd_ia", "rn_Ce_ia", "rn_Ch_ia")
    offset = HEADER_BYTES
    fields: dict[str, np.ndarray] = {}
    for name, code in KINDS[kind]:
        shape = _shape(code, meta)
        count = int(np.prod(shape))
        stop = offset + 8 * count
        if stop > len(raw):
            raise _refuse(f"truncated {name} in {p.name}")
        fields[name] = (
            np.frombuffer(raw, dtype="<f8", count=count, offset=offset)
            .reshape(shape, order="F")
            .copy()
        )
        if not np.isfinite(fields[name]).all():
            raise _refuse(f"non-finite {name} in {p.name}")
        offset = stop
    if offset != len(raw):
        raise _refuse(f"unexpected trailing bytes in {p.name}")
    return {
        "meta": meta,
        "scalars": dict(zip(scalar_names, map(float, scalars), strict=True)),
        "fields": fields,
    }


def _write_synthetic_record(path: Path, kind: int, *, kt: int = 1, kmm: int = 2) -> None:
    jpi, jpj, jpl, h = 12, 10, 5, 2
    ntsi, ntei, ntsj, ntej = h + 1, jpi - h, h + 1, jpj - h
    ni, nj = ntei - ntsi + 1, ntej - ntsj + 1
    meta = np.array(
        [
            kind,
            kt,
            kmm,
            0,
            jpi,
            jpj,
            jpl,
            h,
            ntsi,
            ntei,
            ntsj,
            ntej,
            ni,
            nj,
            len(KINDS[kind]),
            1,
            1,
            1,
            1,
            0,
        ],
        dtype="<i4",
    )
    with path.open("wb") as out:
        out.write(MAGIC)
        out.write(np.array([VERSION], dtype="<i4").tobytes())
        out.write(meta.tobytes())
        out.write(
            np.array(
                [10800.0, 1 / 10800.0, 1026.0, 3991.86795711963, 1.4e-3, 1.4e-3, 1.4e-3],
                dtype="<f8",
            ).tobytes()
        )
        for index, (_name, code) in enumerate(KINDS[kind], start=1):
            shape = _shape(
                code,
                dict(
                    zip(
                        (
                            "kind",
                            "kt",
                            "kmm",
                            "rank",
                            "jpi",
                            "jpj",
                            "jpl",
                            "nn_hls",
                            "ntsi",
                            "ntei",
                            "ntsj",
                            "ntej",
                            "ni",
                            "nj",
                            "nfields",
                            "endian",
                            "nit000",
                            "jpni",
                            "jpnj",
                            "reserved",
                        ),
                        map(int, meta),
                        strict=True,
                    )
                ),
            )
            values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F") + index
            out.write(np.asfortranarray(values).tobytes(order="F"))


def _verify_relative_manifest(
    path: Path,
    root: Path,
    *,
    plant: bool = False,
    expected_paths: set[str] | None = None,
) -> int:
    if not path.is_file():
        raise _refuse(f"missing checksum manifest {path}")
    checked = 0
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise _refuse(f"malformed checksum entry in {path}")
        rel = Path(parts[1].strip())
        if rel.is_absolute() or ".." in rel.parts:
            raise _refuse(f"manifest path is not retained-relative: {rel}")
        rel_name = rel.as_posix()
        if rel_name in seen:
            raise _refuse(f"duplicate checksum path in {path}: {rel_name}")
        seen.add(rel_name)
        target = root / rel
        expected = "0" * 64 if plant and checked == 0 else parts[0]
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise _refuse(f"checksum mismatch for {rel}")
        checked += 1
    if checked == 0:
        raise _refuse(f"empty checksum manifest {path}")
    if expected_paths is not None and seen != expected_paths:
        raise _refuse(
            f"checksum path set differs in {path}: "
            f"missing={sorted(expected_paths - seen)}, extra={sorted(seen - expected_paths)}"
        )
    return checked


def verify_o7_log(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Use O1's whitespace-tolerant real-log parser, then pin production O7."""
    base = verify_ocean_output_log(path, expect_icedyn=True, expect_icethd=True)
    text = Path(path).read_text(errors="replace")
    expected = {
        "ln_icedyn": True,
        "ln_icethd": True,
        "ln_pnd": True,
        "ln_pnd_alb": True,
        "ln_blk": True,
        "ln_mixcpl": False,
        "ln_Cx_ice_cst": True,
    }
    if plant == "selector":
        text = re.sub(r"(ln_pnd\s*=\s*)T", r"\1F", text, count=1)
    parsed: dict[str, bool] = {}
    for name, want in expected.items():
        matches = re.findall(
            rf"\b{re.escape(name)}\s*=\s*(\.?true\.?|\.?false\.?|T|F)\b", text, re.I
        )
        if not matches:
            raise _refuse(f"missing production selector {name}")
        got = matches[-1].strip(".").upper() in {"T", "TRUE"}
        if got is not want:
            raise _refuse(f"production selector {name}={got}, expected {want}")
        parsed[name] = got
    for name, want in (("nn_it000", 1), ("nn_itend", 2), ("nn_fsbc", 1), ("jpni", 1), ("jpnj", 1)):
        matches = re.findall(rf"\b{re.escape(name)}\s*=\s*(-?\d+)", text, re.I)
        if not matches or int(matches[-1]) != want:
            raise _refuse(f"{name} is not {want}")
    return {**base, "o7_production": parsed}


def same_program(
    new_bld: Path,
    ref_bld: Path,
    patches: dict[str, Path],
    writer: Path,
    *,
    new_cfg: str,
    ref_cfg: str,
) -> dict[str, Any]:
    """Compare like with like: strip marked blocks from the real ppsrc files."""
    try:
        report = same_program_observers(
            new_bld,
            ref_bld,
            patches,
            (writer.stem,),
            new_cfg=new_cfg,
            ref_cfg=ref_cfg,
            preprocessed_markers=MARKERS,
        )
    except SameProgramError as exc:
        raise _refuse(str(exc)) from exc
    return {**report, "status": "PASS_SAME_PROGRAM_REAL_PPSRC_MARKERS"}


def _region_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seam = np.zeros_like(mask)
    seam[:2, :] = mask[:2, :]
    seam[-2:, :] = mask[-2:, :]
    fold = np.zeros_like(mask)
    fold[:, -3:] = mask[:, -3:]
    return seam, fold


def _compare(label: str, actual: Any, expected: np.ndarray, mask2: np.ndarray) -> dict[str, Any]:
    got = np.asarray(actual)
    mask = mask2 if got.ndim == 2 else np.broadcast_to(mask2[..., None], got.shape)
    wet = max_ulp_diff(got, expected, mask=mask, strict_signed_zero=True)
    full = max_ulp_diff(got, expected, strict_signed_zero=True)
    norm = max_normalized_diff(got, expected, mask=mask)
    seam2, fold2 = _region_masks(mask2)
    seam = seam2 if got.ndim == 2 else np.broadcast_to(seam2[..., None], got.shape)
    fold = fold2 if got.ndim == 2 else np.broadcast_to(fold2[..., None], got.shape)
    seam_ulp = max_ulp_diff(got, expected, mask=seam, strict_signed_zero=True)
    fold_ulp = max_ulp_diff(got, expected, mask=fold, strict_signed_zero=True)
    if wet or full:
        raise _refuse(f"{label} not exact: wet={wet} full={full} norm={norm}")
    return {
        "max_ulp_wet": wet,
        "max_ulp_full": full,
        "max_norm_wet": norm,
        "seam_max_ulp": seam_ulp,
        "fold_max_ulp": fold_ulp,
    }


def _load_step(records: Path, kt: int, plant: str | None = None) -> dict[int, dict[str, Any]]:
    result = {}
    for kind in KINDS:
        path = records / f"o7_kind{kind:02d}_step{kt:04d}_rank0000.bin"
        if plant == "missing-frame" and kind == max(KINDS):
            path = path.with_name("missing.bin")
        if not path.is_file():
            raise _refuse(f"missing frame {path.name}")
        result[kind] = read_o7_record(path)
    levels = {item["meta"]["kmm"] for item in result.values()}
    if plant == "time-level":
        levels.add(3)
    if levels != {2}:
        raise _refuse(f"SAS Kmm must be Nnn=2, got {levels}")
    if {item["meta"]["kt"] for item in result.values()} != {kt}:
        raise _refuse(f"mixed kt in requested step {kt}")
    if {item["meta"]["nit000"] for item in result.values()} != {1}:
        raise _refuse("record does not derive from production nn_it000=1")
    scalar_sets = {tuple(item["scalars"].items()) for item in result.values()}
    if len(scalar_sets) != 1:
        raise _refuse("run scalars vary within a step")
    if plant == "wet-1ulp":
        mask = result[1]["fields"]["smask0"]
        wi, wj = first_wet_ij(mask, interior_only=False, nn_hls=0)
        field = result[2]["fields"]["wndm_ice"]
        field[wi, wj] = np.nextafter(field[wi, wj], np.inf)
    return result


def _score_step(step: dict[int, dict[str, Any]], config: SI3SurfaceConfig) -> dict[str, Any]:
    f = {kind: item["fields"] for kind, item in step.items()}
    mask = f[1]["smask0"]
    fold = SI3ORCA2FoldProgram(nn_hls=2, nfd_full_when_absent=True)
    scores: dict[str, Any] = {}
    scalars = step[1]["scalars"]
    expected_scalars = {
        "rDt_ice": config.rDt_ice,
        "r1_Dt_ice": 1.0 / config.rDt_ice,
        "rho0": constants.rho_ocean_nemo,
        "rcp": NEMO_SI3_CONSTANTS_CONFIG.c_ocean,
        "rn_Cd_ia": config.rn_Cd_ia,
        "rn_Ce_ia": config.rn_Ce_ia,
        "rn_Ch_ia": config.rn_Ch_ia,
    }
    if scalars != expected_scalars:
        raise _refuse(f"record scalars disagree with resolved deck/constants: {scalars}")

    b1 = jax.jit(lambda **kw: si3_blk_ice_1(config=config, **kw))(
        wndi=jnp.asarray(f[1]["wndi"]),
        wndj=jnp.asarray(f[1]["wndj"]),
        theta_air_zt=jnp.asarray(f[1]["theta_air_zt"]),
        q_air_zt=jnp.asarray(f[1]["q_air_zt"]),
        slp=jnp.asarray(f[1]["slp"]),
        tm_su=jnp.asarray(f[1]["tm_su"]),
        rhoa=jnp.asarray(f[1]["rhoa"]),
    )
    for name in b1._fields:
        scores[f"blk1.{name}"] = _compare(f"blk1.{name}", getattr(b1, name), f[2][name], mask)
    tau = jax.jit(lambda u, v: si3_ice_sbc_tau_lbc(u, v, fold))(b1.utau_ice, b1.vtau_ice)
    full_shape = f[3]["utau_ice"].shape
    if full_shape != (mask.shape[0] + 4, mask.shape[1] + 4):
        raise _refuse(f"interior/full extent mismatch: {mask.shape} -> {full_shape}")
    full_mask = np.pad(mask, ((2, 2), (2, 2)))
    scores["tau.utau_ice"] = _compare("tau.utau_ice", tau[0], f[3]["utau_ice"], full_mask)
    scores["tau.vtau_ice"] = _compare("tau.vtau_ice", tau[1], f[3]["vtau_ice"], full_mask)

    alb = jax.jit(lambda **kw: si3_ice_alb(config=config, **kw))(
        **{
            name: jnp.asarray(f[4][name])
            for name in ("t_su", "h_i", "h_s", "a_ip_eff", "h_ip", "h_il", "cloud_fra", "smask0")
        }
    )
    scores["ice_alb.alb_ice"] = _compare("ice_alb.alb_ice", alb, f[5]["alb_ice"], mask)

    b2_kwargs = {name: jnp.asarray(f[6][name]) for name in f[6] if name not in {"snow", "qlw"}}
    b2_kwargs["snow"] = jnp.asarray(f[6]["snow"])
    b2_kwargs["qlw"] = jnp.asarray(f[6]["qlw"])
    b2 = jax.jit(lambda **kw: si3_blk_ice_2(config=config, **kw))(**b2_kwargs)
    for name in b2._fields:
        scores[f"blk2.{name}"] = _compare(f"blk2.{name}", getattr(b2, name), f[7][name], mask)

    neigh = orca2_flx_other_neighbours(
        u_ice=f[8]["u_ice"], v_ice=f[8]["v_ice"], ssu_m=f[8]["ssu_m"], ssv_m=f[8]["ssv_m"], nn_hls=2
    )
    other_args = {name: jnp.asarray(value) for name, value in neigh.items()}
    other_args.update(
        {
            name: jnp.asarray(f[8][name])
            for name in (
                "utau",
                "vtau",
                "smask0",
                "at_i",
                "at_i_b",
                "vt_i",
                "qsr_oce",
                "qns_oce",
                "qemp_oce",
                "frq_m",
                "e3t_m",
                "sst_m",
                "t_bo",
                "rn_amax_2d",
            )
        }
    )
    other = jax.jit(lambda **kw: si3_ice_flx_other(config=config, **kw))(**other_args)
    for name in other._fields:
        scores[f"flx_other.{name}"] = _compare(
            f"flx_other.{name}", getattr(other, name), f[9][name], mask
        )
    scores["flx_other.frq_m"] = _compare("flx_other.frq_m", f[8]["frq_m"], f[9]["frq_m"], mask)

    composed_args = {
        "wndi": f[1]["wndi"],
        "wndj": f[1]["wndj"],
        "theta_air_zt": f[1]["theta_air_zt"],
        "q_air_zt": f[1]["q_air_zt"],
        "slp": f[1]["slp"],
        "tm_su": f[1]["tm_su"],
        "rhoa": f[1]["rhoa"],
        "t_su": f[4]["t_su"],
        "h_i": f[4]["h_i"],
        "h_s": f[4]["h_s"],
        "a_ip_eff": f[4]["a_ip_eff"],
        "h_ip": f[4]["h_ip"],
        "h_il": f[4]["h_il"],
        "cloud_fra": f[4]["cloud_fra"],
        "smask0": mask,
        "qlw": f[6]["qlw"],
        "precip": f[6]["precip"],
        "snow": f[6]["snow"],
        "qsr": f[6]["qsr"],
        "sst_m": f[6]["sst_m"],
        "emp": f[6]["emp"],
        "at_i_b": f[6]["at_i_b"],
        "a_i_b": f[6]["a_i_b"],
        "qns_oce": f[6]["qns_oce"],
        "qsr_oce": f[6]["qsr_oce"],
        "u_ice": f[8]["u_ice"],
        "v_ice": f[8]["v_ice"],
        "ssu_m": f[8]["ssu_m"],
        "ssv_m": f[8]["ssv_m"],
        "utau": f[8]["utau"],
        "vtau": f[8]["vtau"],
        "at_i": f[8]["at_i"],
        "vt_i": f[8]["vt_i"],
        "frq_m": f[8]["frq_m"],
        "e3t_m": f[8]["e3t_m"],
        "t_bo": f[8]["t_bo"],
        "rn_amax_2d": f[8]["rn_amax_2d"],
    }
    composed = jax.jit(lambda **kw: si3_surface_step(config=config, fold=fold, **kw))(
        **{name: jnp.asarray(value) for name, value in composed_args.items()}
    )
    for name in composed.blk_ice_1._fields:
        scores[f"composed.blk1.{name}"] = _compare(
            f"composed.blk1.{name}", getattr(composed.blk_ice_1, name), f[2][name], mask
        )
    scores["composed.utau"] = _compare(
        "composed.utau", composed.utau_ice, f[3]["utau_ice"], full_mask
    )
    scores["composed.vtau"] = _compare(
        "composed.vtau", composed.vtau_ice, f[3]["vtau_ice"], full_mask
    )
    scores["composed.albedo"] = _compare("composed.albedo", composed.alb_ice, f[5]["alb_ice"], mask)
    for name in composed.blk_ice_2._fields:
        scores[f"composed.blk2.{name}"] = _compare(
            f"composed.blk2.{name}", getattr(composed.blk_ice_2, name), f[7][name], mask
        )
    for name in composed.flx_other._fields:
        scores[f"composed.flx_other.{name}"] = _compare(
            f"composed.flx_other.{name}", getattr(composed.flx_other, name), f[9][name], mask
        )
    return {
        "scores": scores,
        "dtype": str(np.asarray(composed.alb_ice).dtype),
        "execution": "CPU production jit",
        "transcendentals": "libm",
    }


def _completion(run: Path, label: str) -> dict[str, Any]:
    if not (run / "RUN_COMPLETE").is_file():
        raise _refuse(f"missing RUN_COMPLETE in {label}")
    stdout = (run / "run.user.stdout.log").read_text(errors="replace")
    if not re.search(r"STOP\s+0\b", stdout) or re.search(r"\bABORT\b|E\s+R\s+R\s+O\s+R", stdout):
        raise _refuse(f"{label} did not finish with clean STOP 0")
    return verify_o7_log(run / "ocean.output")


def admit_evidence_directory(
    evidence_dir: str | Path,
    *,
    expect_commit: str,
    plant: str | None = None,
) -> dict[str, Any]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    libc_name, libc_version = platform.libc_ver()
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise _refuse("O7 admission requires CPU JAX with x64 enabled")
    if libc_name.lower() != "glibc":
        raise _refuse(f"O7 libm arm requires glibc, got {libc_name!r}")
    ev = Path(evidence_dir).resolve()
    manifest = ev / "manifest"
    if not (ev / "RUN_COMPLETE").is_file():
        raise _refuse("retained evidence lacks top-level RUN_COMPLETE")
    producer = (manifest / "producer_commit.txt").read_text().strip()
    if plant == "stamp":
        producer = "0" * 40
    if (
        producer != expect_commit
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{producer}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    ):
        raise _refuse("producer commit mismatch or missing commit object")
    binary_paths = {"manifest/binaries/nemo.ref", "manifest/binaries/nemo.rec"}
    record_paths = {
        f"record_2step/o7_si3_records/o7_kind{kind:02d}_step{kt:04d}_rank0000.bin"
        for kt in STEPS
        for kind in KINDS
    }
    binaries = _verify_relative_manifest(
        manifest / "binaries.sha256",
        ev,
        plant=plant == "binary-checksum",
        expected_paths=binary_paths,
    )
    records = _verify_relative_manifest(
        manifest / "records.sha256",
        ev,
        plant=plant == "checksum",
        expected_paths=record_paths,
    )
    inputs = _verify_relative_manifest(
        manifest / "inputs.sha256", ev, plant=plant == "input-checksum"
    )
    for name, want in (
        ("same_program.json", "PASS_SAME_PROGRAM_REAL_PPSRC_MARKERS"),
        ("fp_census.json", "PASS_FP_CENSUS_REAL_FLAGS"),
    ):
        report = json.loads((manifest / name).read_text())
        if report.get("status") != want:
            raise _refuse(f"retained {name} did not pass: {report.get('status')!r}")
    ref, rec = ev / "reference_2step", ev / "record_2step"
    ref_log, rec_log = _completion(ref, "reference_2step"), _completion(rec, "record_2step")
    ref_restarts = sorted(p for p in ref.glob("*restart*.nc") if p.is_file())
    if not ref_restarts:
        raise _refuse("reference run has no restart")
    rec_restart_names = {p.name for p in rec.glob("*restart*.nc") if p.is_file()}
    ref_restart_names = {p.name for p in ref_restarts}
    if rec_restart_names != ref_restart_names:
        raise _refuse("reference/record restart file sets differ")
    restart_count = 0
    for ref_path in ref_restarts:
        rec_path = rec / ref_path.name
        left = ref_path.read_bytes()
        right = rec_path.read_bytes() if rec_path.is_file() else b""
        if plant == "restart" and restart_count == 0:
            right += b"x"
        if left != right:
            raise _refuse(f"reference/record restart bytes differ: {ref_path.name}")
        restart_count += 1
    config = SI3SurfaceConfig.from_resolved_namelists(
        rec / "output.namelist.dyn", rec / "output.namelist.ice"
    )
    reference_config = SI3SurfaceConfig.from_resolved_namelists(
        ref / "output.namelist.dyn", ref / "output.namelist.ice"
    )
    if reference_config != config:
        raise _refuse("reference and record resolved O7 configurations differ")
    deck = manifest / "production_deck"
    production = SI3SurfaceConfig.from_input_namelists(
        deck / "namelist_ref",
        deck / "namelist_cfg",
        deck / "namelist_ice_ref",
        deck / "namelist_ice_cfg",
    )
    if config != production:
        raise _refuse("resolved record config is not the unmodified production ORCA2_SAS_ICE deck")
    if not (config.rn_Cd_ia == config.rn_Ce_ia == config.rn_Ch_ia == 1.4e-3):
        raise _refuse("resolved ice-atmosphere coefficients are not all 1.4e-3")
    record_dir = rec / "o7_si3_records"
    expected_records = {
        f"o7_kind{kind:02d}_step{kt:04d}_rank0000.bin" for kt in STEPS for kind in KINDS
    }
    actual_records = {path.name for path in record_dir.iterdir() if path.is_file()}
    if actual_records != expected_records:
        raise _refuse(
            f"record frame set differs: missing={sorted(expected_records - actual_records)}, "
            f"extra={sorted(actual_records - expected_records)}"
        )
    scores, loaded = {}, {}
    for kt in STEPS:
        loaded[kt] = _load_step(record_dir, kt, plant=plant)
        scores[str(kt)] = _score_step(loaded[kt], config)
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    native_fired = False
    try:
        _score_step(loaded[STEPS[0]], config)
    except ValueError:
        native_fired = True
    finally:
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if not native_fired:
        raise _refuse("native-transcendental control was vacuous on the retained record")
    return {
        "status": "PASS_O7_ADMISSION",
        "producer_commit": producer,
        "binaries_checked": binaries,
        "records_checked": records,
        "inputs_checked": inputs,
        "restart_files_byte_identical": restart_count,
        "reference_log": ref_log,
        "record_log": rec_log,
        "steps": scores,
        "native_transcendental_plant": "FIRED",
        "platform": {
            "backend": "cpu",
            "dtype": "float64",
            "libc": libc_name,
            "libc_version": libc_version,
            "transcendentals": "libm",
        },
    }


def run_self_test(plant: str | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for kind in KINDS:
            _write_synthetic_record(root / f"o7_kind{kind:02d}_step0001_rank0000.bin", kind)
        path = root / "o7_kind01_step0001_rank0000.bin"
        rec = read_o7_record(
            path, plant=plant if plant in {"header", "truncation", "extent"} else None
        )
        if plant in {"time-level", "missing-frame"}:
            _load_step(root, 1, plant=plant)
        rel = Path("o7_kind01_step0001_rank0000.bin")
        manifest = root / "records.sha256"
        manifest.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}\n")
        _verify_relative_manifest(manifest, root, plant=plant == "checksum")
        mask = np.ones((8, 6))
        mask[:2, :] = 0.0
        mask[3, 3] = 0.0
        signed = verify_dry_signed_zero_fixture(np.zeros((8, 6)), mask, nn_hls=2)
        wi, wj = first_wet_ij(mask, nn_hls=2)
        base = np.ones((8, 6))
        changed = base.copy()
        changed[wi, wj] = np.nextafter(1.0, np.inf)
        if plant == "wet-1ulp" and max_ulp_diff(base, changed, mask=mask) == 1:
            raise _refuse("wet one-ULP plant refused")
        if plant == "absolute-path":
            manifest.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
            _verify_relative_manifest(manifest, root)
        return {
            "status": "PASS_SELF_TEST",
            "header_bytes": HEADER_BYTES,
            "field_count": len(rec["fields"]),
            "signed_zero": signed,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-record")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--verify-log")
    mode.add_argument("--same-program")
    mode.add_argument("--admit-evidence")
    parser.add_argument("--plant")
    parser.add_argument("--expect-commit")
    parser.add_argument("--reference-bld")
    parser.add_argument("--writer")
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--new-cfg", default="ORCA2_SAS_ICE_O7_REC")
    parser.add_argument("--reference-cfg", default="ORCA2_SAS_ICE_O7_REF")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    try:
        if args.read_record:
            rec = read_o7_record(args.read_record, args.plant)
            result = {
                "meta": rec["meta"],
                "fields": {k: list(v.shape) for k, v in rec["fields"].items()},
            }
        elif args.self_test:
            result = run_self_test(args.plant)
        elif args.verify_log:
            result = verify_o7_log(args.verify_log, args.plant)
        elif args.same_program:
            if not args.reference_bld or not args.writer or len(args.patch) != len(OBSERVER_UNITS):
                raise _refuse("same-program requires reference, writer, and four unit=patch pairs")
            patches = {}
            for item in args.patch:
                unit, sep, path = item.partition("=")
                if not sep or unit not in OBSERVER_UNITS:
                    raise _refuse(f"bad --patch {item}")
                patches[unit] = Path(path)
            if set(patches) != set(OBSERVER_UNITS):
                raise _refuse("same-program patch unit set incomplete")
            result = same_program(
                Path(args.same_program),
                Path(args.reference_bld),
                patches,
                Path(args.writer),
                new_cfg=args.new_cfg,
                ref_cfg=args.reference_cfg,
            )
        else:
            if not args.expect_commit:
                raise _refuse("--expect-commit required")
            result = admit_evidence_directory(
                args.admit_evidence, expect_commit=args.expect_commit, plant=args.plant
            )
    except (ValueError, OSError, KeyError) as exc:
        print(exc)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        Path(args.json).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
