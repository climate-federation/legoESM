#!/usr/bin/env python3
"""Score the preregistered row-1.3 QCO continuity operand ladder.

The NEMO writer is direct and write-only.  This scorer calls legoESM's shipped
face-depth and divergence functions on the independently dumped, bit-exact
substep-1 eta/velocity inputs, then applies the nested Q/F/B oracle holds from
``PREREG_split_explicit_momentum_chain_round9.md``.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import nemo_ssh_avg_face_depth
from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo

import split_explicit_momentum_chain_round1 as base
import spg_substep_chain as inherited
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    sha256,
    stream_manifest,
)


EXPECTED_NEW = {
    "qco_dump_zsshp2_substep1.bin",
    "qco_dump_zhup2_substep1.bin",
    "qco_dump_zhvp2_substep1.bin",
    "qco_dump_zhU_substep1.bin",
    "qco_dump_zhV_substep1.bin",
    "qco_dump_zhdiv_substep1.bin",
}
POINTWISE = 1.0e-15
ACCUMULATING = 1.0e-12


def _u_face(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[:, -1:], a], axis=1)


def _v_face(a: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros_like(a[:1]), a], axis=0)


def _u_nemo(a: np.ndarray) -> np.ndarray:
    return np.asarray(a)[:, 1:]


def _v_nemo(a: np.ndarray) -> np.ndarray:
    return np.asarray(a)[1:, :]


def _raw_record(
    subrow: str,
    field: str,
    lego: np.ndarray,
    nemo: np.ndarray,
    mask: np.ndarray,
    population: int,
    arithmetic_class: str,
    class_bar: float,
) -> dict[str, Any]:
    metric = base._metric(lego, nemo, mask, population)
    at_bar = (
        metric["correlation"] >= 1.0 - 1.0e-9
        and abs(metric["mean_abs_ratio"] - 1.0) <= 1.0e-6
        and metric["per_element_max_error_over_nemo_rms"] <= class_bar
    )
    return {
        "subrow": subrow,
        "field": field,
        "arithmetic_class": arithmetic_class,
        "class_bar": class_bar,
        **metric,
        "gate_status": "AT BAR" if at_bar else "DEBT",
    }


def _prediction(prediction: np.ndarray, residual: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    wet = np.asarray(mask, dtype=bool)
    p = np.asarray(prediction)[wet]
    r = np.asarray(residual)[wet]
    rms = float(np.sqrt(np.mean(r * r)))
    if not math.isfinite(rms) or rms == 0.0:
        raise RuntimeError("registered continuity residual has zero/non-finite RMS")
    error = float(np.sqrt(np.mean((r - p) ** 2))) / rms
    explained = 1.0 - error
    corr = float(np.corrcoef(p, r)[0, 1]) if np.std(p) > 0.0 else 0.0
    if error <= 0.10 and explained >= 0.90 and corr >= 0.99:
        verdict = "CONFIRMS"
    elif error >= 0.90 and explained <= 0.10 and corr <= 0.20:
        verdict = "REFUTES"
    else:
        verdict = "UNRESOLVED"
    return {
        "n": int(wet.sum()),
        "prediction_normalized_error": error,
        "explained_rms_fraction": explained,
        "correlation": corr,
        "verdict": verdict,
    }


def _wall_pointwise(
    prediction: np.ndarray, residual: np.ndarray, land: np.ndarray, row: int
) -> dict[str, Any]:
    cols = np.flatnonzero(land[row])
    delta = np.asarray(prediction)[row, cols]
    target = np.asarray(residual)[row, cols]
    error = target - delta
    k = int(np.argmax(np.abs(error)))
    return {
        "row": row,
        "columns": cols.tolist(),
        "prediction": delta.tolist(),
        "residual": target.tolist(),
        "residual_minus_prediction": error.tolist(),
        "argmax_abs_error": {
            "column": int(cols[k]),
            "prediction": float(delta[k]),
            "residual": float(target[k]),
            "residual_minus_prediction": float(error[k]),
        },
    }


def _bound_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA mismatch: expected {expected}, got {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--off-run", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--off-binary-sha256", required=True)
    parser.add_argument("--dynspg-source-sha256", required=True)
    parser.add_argument("--dump-sha", action="append", default=[])
    parser.add_argument("--bracket-receipt", type=Path, required=True)
    parser.add_argument("--bracket-sha256", required=True)
    parser.add_argument("--round8-artifact", type=Path, required=True)
    parser.add_argument("--round8-sha256", required=True)
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("DINO_1226_LANE") != "d180":
        raise SystemExit("DINO_1226_LANE=d180 is required")
    if os.environ.get("LEGOESM_NEMO_E3T") != "both":
        raise SystemExit("LEGOESM_NEMO_E3T=both is required")
    session_id = os.environ.get("CODEX_SESSION_ID", "")
    if not session_id or session_id == "unset":
        raise SystemExit("non-empty CODEX_SESSION_ID is required")
    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX fp64 are required")

    root = Path(__file__).resolve().parents[4]
    production_paths = {
        "face_depth": Path(inspect.getsourcefile(nemo_ssh_avg_face_depth) or "").resolve(),
        "divergence": Path(inspect.getsourcefile(divergence_cgrid) or "").resolve(),
        "bridge": Path(inspect.getsourcefile(bridge_nemo_to_legoesm_topo) or "").resolve(),
    }
    escaped = {k: str(v) for k, v in production_paths.items() if not v.is_relative_to(root)}
    if escaped:
        raise SystemExit(f"production imports escaped measured checkout: {escaped}")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if git_sha != args.package_commit:
        raise SystemExit(
            f"measured checkout {git_sha} != frozen package commit {args.package_commit}"
        )
    dirty_before = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if dirty_before:
        raise SystemExit("clean worktree required\n" + dirty_before)

    run = args.run.resolve()
    frozen_off = args.off_run.resolve()
    _bound_hash(run / "nemo", args.binary_sha256, "NEMO executable")
    _bound_hash(frozen_off / "nemo", args.off_binary_sha256, "OFF NEMO executable")
    if (run / ".nemo_binary_sha256").read_text().split()[0] != args.binary_sha256:
        raise SystemExit("run-stamped executable SHA mismatch")
    if (frozen_off / ".nemo_binary_sha256").read_text().split()[0] != args.off_binary_sha256:
        raise SystemExit("OFF run-stamped executable SHA mismatch")
    if (run / ".dynspg_source_sha256").read_text().split()[0] != args.dynspg_source_sha256:
        raise SystemExit("run-stamped dynspg source SHA mismatch")
    _bound_hash(args.bracket_receipt, args.bracket_sha256, "bracket receipt")
    _bound_hash(args.round8_artifact, args.round8_sha256, "round-8 artifact")

    supplied: dict[str, str] = {}
    for binding in args.dump_sha:
        name, sep, value = binding.partition("=")
        if not sep or name in supplied:
            raise SystemExit(f"invalid/duplicate --dump-sha binding: {binding!r}")
        supplied[name] = value
    if set(supplied) != EXPECTED_NEW:
        raise SystemExit(f"dump SHA set mismatch: {set(supplied) ^ EXPECTED_NEW}")
    for name, value in supplied.items():
        _bound_hash(run / name, value, name)

    bracket = json.loads(args.bracket_receipt.read_text())
    if not (
        bracket.get("schema") == "dino-spg-qco-row13-bracket-v1"
        and Path(bracket.get("on_dir", "")).resolve() == run
        and Path(bracket.get("off_dir", "")).resolve() == frozen_off
        and bracket.get("shared_count") == 197
        and bracket.get("shared_exact") is True
        and set(bracket.get("new_streams", [])) == EXPECTED_NEW
        and all(bracket.get("controls", {}).values())
    ):
        raise SystemExit("invalid QCO ON/OFF bracket receipt")
    on_manifest = stream_manifest(run)
    off_manifest = stream_manifest(frozen_off)
    if not (
        len(on_manifest) == 203
        and len(off_manifest) == 197
        and set(on_manifest) - set(off_manifest) == EXPECTED_NEW
        and all(files_byte_identical(run / name, frozen_off / name) for name in off_manifest)
    ):
        raise SystemExit("scored ON directory no longer satisfies exact 197/197 bracket")
    shared_hash = manifest_sha256({name: on_manifest[name] for name in sorted(off_manifest)})
    if shared_hash != bracket.get("shared_manifest_sha256"):
        raise SystemExit("scored shared-stream manifest changed after bracketing")
    round8 = json.loads(args.round8_artifact.read_text())
    continuity8 = round8["continuity_localization"]
    if continuity8["status"] != "CONFIRMED_CONTINUITY_FLUX_COMPOSITION":
        raise SystemExit("round-8 continuity stop is not confirmed")
    if continuity8["midstep_u"]["gate_status"] != "AT BAR" or continuity8["midstep_v"]["gate_status"] != "AT BAR":
        raise SystemExit("round-8 half-step velocities are not AT BAR")
    dt_e = float(continuity8["dt_s"])

    jpi, jpj, _jpk, hls, _icycle, _nn_e = inherited._read_dims(str(frozen_off))
    if (jpi, jpj, hls) != (56, 203, 2):
        raise SystemExit(f"unexpected DINO dimensions {(jpi, jpj, hls)}")
    for name in EXPECTED_NEW:
        if (run / name).stat().st_size != 90944:
            raise SystemExit(f"{name}: expected 90,944 bytes")

    load = lambda name: inherited._load_full(str(run / name), jpi, jpj, hls)
    eta_seed = load("spg_dump_sshn_e_init.bin")
    ua = load("cor2d_dump_ua_e_in_substep1.bin")
    va = load("cor2d_dump_va_e_in_substep1.bin")
    ssh_out = load("spg_dump_ssh_substep1.bin")
    ssh_frc = load("spg_dump_ssh_frc.bin")
    n_zsshp = load("qco_dump_zsshp2_substep1.bin")
    n_hu = load("qco_dump_zhup2_substep1.bin")
    n_hv = load("qco_dump_zhvp2_substep1.bin")
    n_zhU = load("qco_dump_zhU_substep1.bin")
    n_zhV = load("qco_dump_zhV_substep1.bin")
    n_div = load("qco_dump_zhdiv_substep1.bin")

    legacy_round8 = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
        "cfgs/DINO/RUN_SEQDUMP_D180_1R"
    ).resolve()
    frozen_dump_names = {
        "spg_dump_sshn_e_init.bin",
        "cor2d_dump_ua_e_in_substep1.bin",
        "cor2d_dump_va_e_in_substep1.bin",
        "spg_dump_ssh_substep1.bin",
        "spg_dump_ssh_frc.bin",
    }
    for name in frozen_dump_names:
        _bound_hash(
            legacy_round8 / name,
            round8["dump_sha256"][name],
            f"legacy round-8 {name}",
        )
    for name in ("mesh_mask.nc", inherited.RESTART_FILE):
        if sha256(run / name) != round8["input_sha256"][name]:
            raise SystemExit(f"{name}: differs from frozen round-8 input")

    g = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(str(run / inherited.RESTART_FILE), nn_hls=0)
    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0,
        lon_east_deg=49.0,
        sill_lon_m_deg=1.0,
    )
    bridge_kw: dict[str, Any] = {
        "periodic_i": True,
        "full_step": True,
        "omega": cfg.omega,
    }
    if "carry_native_lat_deg" in inspect.signature(bridge_nemo_to_legoesm_topo).parameters:
        bridge_kw["carry_native_lat_deg"] = (
            getattr(cfg, "tke_htau_evaluation", None) == "nemo_literal"
            or getattr(cfg, "gm_treguier_final_evaluation", None) == "nemo_literal"
        )
    br = bridge_nemo_to_legoesm_topo(g, s, **bridge_kw)
    land = np.asarray(g.tmask)[..., 0] > 0.5
    umask = np.asarray(g.umask)[..., 0] > 0.5
    vmask = np.asarray(g.vmask)[..., 0] > 0.5

    # The deterministic writer deliberately normalizes halo bytes, so its
    # full-file hashes cannot equal the legacy round-8 files. Bind those old
    # files to their frozen receipt, then require bit identity on each field's
    # registered wet physical population. A one-ULP wet-cell plant proves this
    # translation gate can fail.
    fresh_frozen = {
        "spg_dump_sshn_e_init.bin": eta_seed,
        "cor2d_dump_ua_e_in_substep1.bin": ua,
        "cor2d_dump_va_e_in_substep1.bin": va,
        "spg_dump_ssh_substep1.bin": ssh_out,
        "spg_dump_ssh_frc.bin": ssh_frc,
    }
    frozen_populations = {
        "spg_dump_sshn_e_init.bin": (land, 9920, "wet_T"),
        "cor2d_dump_ua_e_in_substep1.bin": (umask, 9758, "wet_U"),
        "cor2d_dump_va_e_in_substep1.bin": (vmask, 9868, "wet_V"),
        "spg_dump_ssh_substep1.bin": (land, 9920, "wet_T"),
        "spg_dump_ssh_frc.bin": (land, 9920, "wet_T"),
    }

    def physical_bits_equal(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
        aa = np.ascontiguousarray(np.asarray(a)[mask], dtype=np.float64).view(np.uint64)
        bb = np.ascontiguousarray(np.asarray(b)[mask], dtype=np.float64).view(np.uint64)
        return bool(np.array_equal(aa, bb))

    legacy_load = lambda name: inherited._load_full(
        str(legacy_round8 / name), jpi, jpj, hls)
    deterministic_translation: dict[str, Any] = {}
    translation_plants: list[bool] = []
    for name, fresh in fresh_frozen.items():
        mask, expected_n, population = frozen_populations[name]
        if int(mask.sum()) != expected_n:
            raise SystemExit(f"{name}: {population} population changed")
        legacy = legacy_load(name)
        exact = physical_bits_equal(fresh, legacy, mask)
        if not exact:
            raise SystemExit(f"{name}: deterministic/legacy {population} bits differ")
        planted = np.array(fresh, copy=True)
        first = tuple(np.argwhere(mask)[0])
        if not np.isfinite(planted[first]):
            raise SystemExit(f"{name}: first planted {population} cell is non-finite")
        planted[first] = np.nextafter(planted[first], np.inf)
        plant_fails = not physical_bits_equal(planted, legacy, mask)
        translation_plants.append(plant_fails)
        deterministic_translation[name] = {
            "population": population,
            "n": expected_n,
            "legacy_full_file_sha256": round8["dump_sha256"][name],
            "deterministic_full_file_sha256": sha256(run / name),
            "physical_bits_exact": exact,
            "one_ulp_physical_plant_fails": plant_fails,
        }
    if np.count_nonzero(ssh_frc[land]) != 0:
        raise SystemExit("frozen substep-1 ssh_frc is not exact zero")
    uface_mask = _u_face(umask.astype(np.float64))
    vface_mask = _v_face(vmask.astype(np.float64))
    uface = _u_face(ua)
    vface = _v_face(va)

    prod_hu, prod_hv = nemo_ssh_avg_face_depth(
        jnp.asarray(eta_seed),
        br.state.H_bathy.data,
        jnp.asarray(land),
        jnp.asarray(uface_mask),
        jnp.asarray(vface_mask),
        br.geometry,
        br.geometry.area,
        jnp.float64,
    )
    prod_hu = np.asarray(prod_hu)
    prod_hv = np.asarray(prod_hv)
    base_fu = prod_hu * uface * uface_mask
    base_fv = prod_hv * vface * vface_mask
    base_div = np.asarray(divergence_cgrid(
        jnp.asarray(base_fu), jnp.asarray(base_fv), br.geometry,
        u_mask=jnp.asarray(uface_mask), v_mask=jnp.asarray(vface_mask),
    ))

    # Q: direct oracle depth, then unchanged production product/divergence.
    q_fu = _u_face(n_hu) * uface * uface_mask
    q_fv = _v_face(n_hv) * vface * vface_mask
    q_div = np.asarray(divergence_cgrid(
        jnp.asarray(q_fu), jnp.asarray(q_fv), br.geometry,
        u_mask=jnp.asarray(uface_mask), v_mask=jnp.asarray(vface_mask),
    ))
    # F: direct oracle metric flux, then unchanged production divergence.  The
    # injected velocity-like fields are divided by the exact widths that
    # divergence_cgrid will multiply back in.  Dividing by NEMO e2u/e1v here
    # would silently retain a geometry ratio and would not actually hold zhU/V.
    prod_e2u = np.broadcast_to(
        (np.asarray(br.geometry.dy) * 0.5)[:, None], n_zhU.shape)
    prod_e1v = np.asarray(br.geometry.dx_v)[1:, :]
    if prod_e1v.shape != n_zhV.shape:
        raise SystemExit(f"production V-width shape {prod_e1v.shape} != {n_zhV.shape}")
    f_u_nemo = np.divide(
        n_zhU, prod_e2u, out=np.zeros_like(n_zhU), where=prod_e2u != 0.0)
    f_v_nemo = np.divide(
        n_zhV, prod_e1v, out=np.zeros_like(n_zhV), where=prod_e1v != 0.0)
    f_fu = _u_face(f_u_nemo) * uface_mask
    f_fv = _v_face(f_v_nemo) * vface_mask
    f_div = np.asarray(divergence_cgrid(
        jnp.asarray(f_fu), jnp.asarray(f_fv), br.geometry,
        u_mask=jnp.asarray(uface_mask), v_mask=jnp.asarray(vface_mask),
    ))

    lego_zhU = _u_nemo(base_fu) * prod_e2u
    lego_zhV = _v_nemo(base_fv) * prod_e1v
    measurements = [
        _raw_record("9.1", "zsshp2_e", eta_seed, n_zsshp, land, 9920, "POINTWISE", POINTWISE),
        _raw_record("9.2", "zhup2_e", _u_nemo(prod_hu), n_hu, umask, 9758, "POINTWISE", POINTWISE),
        _raw_record("9.3", "zhvp2_e", _v_nemo(prod_hv), n_hv, vmask, 9868, "POINTWISE", POINTWISE),
        _raw_record("9.4", "zhU", lego_zhU, n_zhU, umask, 9758, "POINTWISE", POINTWISE),
        _raw_record("9.5", "zhV", lego_zhV, n_zhV, vmask, 9868, "POINTWISE", POINTWISE),
        _raw_record("9.6", "zhdiv", base_div, n_div, land, 9920, "ACCUMULATING", ACCUMULATING),
        _raw_record("9.7", "ssha_e", eta_seed - dt_e * (ssh_frc + base_div), ssh_out, land, 9920, "ACCUMULATING", ACCUMULATING),
    ]
    first_debt = next((m["subrow"] for m in measurements if m["gate_status"] != "AT BAR"), None)

    residual = n_div - base_div
    row_masks = {
        "whole_wet_domain": land,
        "j=1": land & (np.arange(land.shape[0])[:, None] == 1),
        "j=197": land & (np.arange(land.shape[0])[:, None] == 197),
    }
    arms = {"Q_depth_held": q_div, "F_flux_held": f_div, "B_nemo_divergence": n_div}
    substitutions = {
        arm: {region: _prediction(value - base_div, residual, mask)
              for region, mask in row_masks.items()}
        for arm, value in arms.items()
    }
    wall_pointwise = {
        arm: {f"j={row}": _wall_pointwise(value - base_div, residual, land, row)
              for row in (1, 197)}
        for arm, value in arms.items()
    }

    oracle_u_identity = _raw_record(
        "control", "oracle_zhU_identity", n_zhU, n_zhU, umask, 9758,
        "POINTWISE", POINTWISE)
    oracle_v_identity = _raw_record(
        "control", "oracle_zhV_identity", n_zhV, n_zhV, vmask, 9868,
        "POINTWISE", POINTWISE)
    shifted_u = np.roll(n_zhU, 1, axis=1)
    flipped_v = -n_zhV
    controls = {
        "oracle_u_identity_at_bar": oracle_u_identity["gate_status"] == "AT BAR",
        "oracle_v_identity_at_bar": oracle_v_identity["gate_status"] == "AT BAR",
        "planted_u_halo_offset_fails": _raw_record(
            "control", "shifted_zhU", shifted_u, n_zhU, umask, 9758,
            "POINTWISE", POINTWISE)["gate_status"] == "DEBT",
        "planted_v_vector_sign_fails": _raw_record(
            "control", "sign_flipped_zhV", flipped_v, n_zhV, vmask, 9868,
            "POINTWISE", POINTWISE)["gate_status"] == "DEBT",
        "zero_prediction_does_not_confirm": _prediction(
            np.zeros_like(residual), residual, land)["verdict"] != "CONFIRMS",
        "bracket_one_bit_and_missing_stream": all(bracket["controls"].values()),
        "deterministic_translation_exact": all(
            row["physical_bits_exact"] for row in deterministic_translation.values()),
        "deterministic_translation_one_ulp_plants_fail": all(translation_plants),
    }
    if not all(controls.values()):
        raise SystemExit(f"planted control failed: {controls}")

    qv = substitutions["Q_depth_held"]["whole_wet_domain"]["verdict"]
    fv = substitutions["F_flux_held"]["whole_wet_domain"]["verdict"]
    bv = substitutions["B_nemo_divergence"]["whole_wet_domain"]["verdict"]
    if first_debt is None:
        disposition = "ALL_ROW13_OPERANDS_AT_BAR"
    elif first_debt == "9.1":
        disposition = "ORDERED_STOP_AT_ZSSHP2_OPEN_UNRESOLVED"
    elif first_debt in {"9.2", "9.3"} and qv == "CONFIRMS":
        disposition = "QCO_FACE_DEPTH_WEIGHTING_OWNS"
    elif first_debt in {"9.4", "9.5"} and qv == "REFUTES" and fv == "CONFIRMS":
        disposition = "FLUX_PRODUCT_COMPOSITION_OWNS"
    elif (first_debt == "9.6" and qv == "REFUTES"
          and fv == "REFUTES" and bv == "CONFIRMS"):
        disposition = "DIVERGENCE_BOUNDARY_COMPOSITION_OWNS_NOT_NARROWER"
    elif first_debt == "9.7":
        disposition = "SSH_UPDATE_COMPOSITION_OPEN_UNRESOLVED"
    else:
        disposition = "OPEN_UNRESOLVED"

    dirty_after = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if dirty_after:
        raise SystemExit("worktree changed during score")
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round9-v1",
        "session_id": session_id,
        "git": {"commit": git_sha, "clean_before": True, "clean_after": True},
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                    "numpy": np.__version__},
        "source_bindings": {
            "binary_sha256": args.binary_sha256,
            "off_binary_sha256": args.off_binary_sha256,
            "dynspg_source_sha256": args.dynspg_source_sha256,
            "package_commit": args.package_commit,
            "round8_artifact_sha256": args.round8_sha256,
            "bracket_receipt_sha256": args.bracket_sha256,
            "production_sha256": {k: sha256(v) for k, v in production_paths.items()},
        },
        "dump_sha256": supplied,
        "deterministic_writer_translation": deterministic_translation,
        "dt_e_s": dt_e,
        "measurements": measurements,
        "first_diverged_subrow": first_debt,
        "substitution_arms": substitutions,
        "wall_pointwise_deltas": wall_pointwise,
        "controls": controls,
        "disposition": disposition,
        "later_rows": {
            "1.4": "ORDERED-BLOCKED",
            **{str(i): "ORDERED-BLOCKED" for i in range(2, 7)},
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND9 first_diverged_subrow={first_debt} disposition={disposition}")
    for arm, rows in wall_pointwise.items():
        for row, detail in rows.items():
            print("WALL_POINTWISE " + json.dumps(
                {"arm": arm, "row": row, **detail}, sort_keys=True))
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
