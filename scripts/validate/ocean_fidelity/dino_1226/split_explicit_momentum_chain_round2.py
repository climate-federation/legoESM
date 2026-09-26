#!/usr/bin/env python3
"""Day-180 row-1.1 peel: decompose F_slow/zu_frc by tendency term.

The term ladder and attribution bars are frozen in
``PREREG_split_explicit_momentum_chain_round2.md``.  This probe consumes only
existing NEMO dumps.  It runs the shipped legacy bridge entry, the basin
lane's faithful T-point prior-stress counterfactual, and one donor-equalized
control; it never edits or executes NEMO.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import jaxlib
import netCDF4 as nc
import numpy as np

import fidelity_bar_gate as bar_gate
import split_explicit_momentum_chain_round1 as round1
import spg_substep_chain as inherited
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.dynamics.barotropic_latlon_cgrid as barotropic_module
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model_module
import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe_module
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_bottom_drag_rate_faces,
)
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
import legoesm.ocean.experiments.dino as dino_module
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)
import legoesm.ocean.fidelity.nemo_state_bridge as bridge_module
from legoesm.ocean.fidelity.precision_gate import require_fp64
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.vertical import compute_layer_thickness


EXPECTED_LANE = "d180"
EXPECTED_U = 9758
EXPECTED_V = 9868
DT = 2700.0
BASE_RECONSTRUCTION_BAR = 1.0e-12
TERM_CLOSURE_BAR = 1.0e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _u_to_nemo(a: np.ndarray) -> np.ndarray:
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a: np.ndarray) -> np.ndarray:
    return np.asarray(a)[1:, :]


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def _restart_clock(path: Path) -> dict[str, float]:
    with nc.Dataset(path) as dataset:
        adatrj = float(np.asarray(dataset.variables["adatrj"][:]).ravel()[0])
        kt = float(np.asarray(dataset.variables["kt"][:]).ravel()[0])
    from_days = adatrj * 86400.0
    from_kt = kt * DT
    if not np.isfinite(from_days) or abs(from_days - from_kt) > DT:
        raise SystemExit(
            f"restart clock mismatch: adatrj={from_days} seconds, kt={from_kt}"
        )
    return {"adatrj_days": adatrj, "kt": kt, "seconds": from_days}


def _oracle_echo_flags(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    text = path.read_text(errors="replace")
    out: dict[str, str] = {}
    for name in names:
        hits = re.findall(rf"\b{re.escape(name)}\b\s*=\s*([TF])\b", text)
        if not hits or len(set(hits)) != 1:
            raise SystemExit(
                f"oracle echo for {name} missing or inconsistent in {path}: {hits}"
            )
        out[name] = hits[0]
    return out


def _register_dumps(kt: int) -> list[str]:
    names_and_sources = {
        "keg_dump_du.bin": "cfgs/DINO/MY_SRC/dynadv.F90:89-95 KEG Krhs contribution",
        "keg_dump_dv.bin": "same as keg_dump_du",
        "zad_dump_du.bin": "cfgs/DINO/MY_SRC/dynadv.F90:97-103 ZAD increment",
        "zad_dump_dv.bin": "same as zad_dump_du",
        "vor_dump_du.bin": "cfgs/DINO/MY_SRC/dynvor.F90:147-193 total EEN increment and dump",
        "vor_dump_dv.bin": "same as vor_dump_du",
        "ldf_dump_du.bin": "cfgs/DINO/MY_SRC/dynldf.F90:69-119 increment and dump",
        "ldf_dump_dv.bin": "same as ldf_dump_du",
        "hpg_dump_du.bin": "cfgs/DINO/MY_SRC/dynhpg.F90:348-413 SCO increment",
        "hpg_dump_dv.bin": "same as hpg_dump_du",
        "drg_dump_zu_frc_inc.bin": "cfgs/DINO/MY_SRC/dynspg_ts.F90:372-400",
        "drg_dump_zv_frc_inc.bin": "same as drg_dump_zu_frc_inc",
        "wnd_dump_zu_frc_inc.bin": "cfgs/DINO/MY_SRC/dynspg_ts.F90:423-459",
        "wnd_dump_zv_frc_inc.bin": "same as wnd_dump_zu_frc_inc",
        "spg_dump_zu_frc.bin": "cfgs/DINO/MY_SRC/dynspg_ts.F90:507-525",
        "spg_dump_zv_frc.bin": "same as spg_dump_zu_frc",
        f"stp_dump_06_dynhpg_kt{kt:08d}_du.bin": (
            "cfgs/DINO/MY_SRC/stpmlf.F90:324-328 accumulated Krhs"
        ),
        f"stp_dump_06_dynhpg_kt{kt:08d}_dv.bin": "same as dynhpg du",
    }
    for name, source in names_and_sources.items():
        register_dump(name, "now", source)
        time_level_for_dump(name)
    return list(names_and_sources)


def _load_terms(run_dir: Path, jpi: int, jpj: int, jpk: int, hls: int, kt: int):
    load3 = lambda name: inherited._load_full_3d(
        str(run_dir / name), jpi, jpj, jpk - 1, hls
    )
    load2 = lambda name: inherited._load_interior(
        str(run_dir / name), jpi - 2 * hls, jpj - 2 * hls
    )
    terms = {
        "u": {
            "keg": load3("keg_dump_du.bin"),
            "vertical_advection": load3("zad_dump_du.bin"),
            "vorticity": load3("vor_dump_du.bin"),
            "lateral_friction": load3("ldf_dump_du.bin"),
            "hpg": load3("hpg_dump_du.bin"),
            "stage06": load3(f"stp_dump_06_dynhpg_kt{kt:08d}_du.bin"),
            "drag": load2("drg_dump_zu_frc_inc.bin"),
            "wind": load2("wnd_dump_zu_frc_inc.bin"),
            "final": load2("spg_dump_zu_frc.bin"),
        },
        "v": {
            "keg": load3("keg_dump_dv.bin"),
            "vertical_advection": load3("zad_dump_dv.bin"),
            "vorticity": load3("vor_dump_dv.bin"),
            "lateral_friction": load3("ldf_dump_dv.bin"),
            "hpg": load3("hpg_dump_dv.bin"),
            "stage06": load3(f"stp_dump_06_dynhpg_kt{kt:08d}_dv.bin"),
            "drag": load2("drg_dump_zv_frc_inc.bin"),
            "wind": load2("wnd_dump_zv_frc_inc.bin"),
            "final": load2("spg_dump_zv_frc.bin"),
        },
    }
    return terms


def _nemo_depth_mean(term, e3, mask3, column_depth):
    nlev = min(term.shape[-1], e3.shape[-1], mask3.shape[-1])
    numerator = np.sum(
        term[..., :nlev] * e3[..., :nlev] * mask3[..., :nlev], axis=-1
    )
    return numerator / np.maximum(column_depth, 1.0e-10)


def _live_depth_mean(term, thickness, mask2):
    nlev = min(term.shape[-1], thickness.shape[-1])
    numerator = np.sum(term[..., :nlev] * thickness[..., :nlev], axis=-1)
    denominator = np.maximum(np.sum(thickness[..., :nlev], axis=-1), 1.0e-10)
    return numerator / denominator * mask2


def _drag_contribution(state, h_k, h_u, h_v, z_coord, config, grid):
    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), 1.0e-10)
    H_v = jnp.maximum(jnp.sum(h_v, axis=-1), 1.0e-10)
    r_u, r_v, isb_u, isb_v = nemo_bottom_drag_rate_faces(
        state.u.data, state.v.data, h_k, z_coord, config, grid
    )
    centred = (
        bool(config.barotropic_forcing_centred)
        and state.u_before is not None
        and state.v_before is not None
    )
    u_src = state.u_before.data if centred else state.u.data
    v_src = state.v_before.data if centred else state.v.data
    u_bot = jnp.sum(u_src * isb_u, axis=-1)
    v_bot = jnp.sum(v_src * isb_v, axis=-1)
    U_bar = jnp.sum(u_src * h_u, axis=-1) / H_u
    V_bar = jnp.sum(v_src * h_v, axis=-1) / H_v
    drag_u = -r_u.astype(u_src.dtype) / H_u * (u_bot - U_bar)
    drag_v = -r_v.astype(v_src.dtype) / H_v * (v_bot - V_bar)
    return np.asarray(drag_u), np.asarray(drag_v)


def _run_arm(state, sf, geometry, z_coord, config) -> dict[str, Any]:
    model = LatLonCGridOceanModel(geometry, z_coord, config)
    captured: dict[str, Any] = {}
    before_fields = (
        state.T_before, state.S_before, state.u_before, state.v_before,
    )
    if any(field is None for field in before_fields):
        raise SystemExit("NEMO leapfrog BEFORE fields required for term peel")
    centred_sf = sf._replace(
        tau_x=0.5 * (state.tau_x_prev + sf.tau_x),
        tau_y=0.5 * (state.tau_y_prev + sf.tau_y),
    )
    real_gradients = pe_module._bc_ke_and_pressure_gradients

    def spy_gradients(*args, **kwargs):
        result = real_gradients(*args, **kwargs)
        if "keg_u" not in captured:
            dke_dx, dp_dx, dke_dy, dp_dy = result
            rho_0 = float(config.constants.rho_0)
            captured["keg_u"] = np.asarray(-dke_dx)
            captured["keg_v"] = np.asarray(-dke_dy)
            captured["hpg_u"] = np.asarray(-dp_dx / rho_0)
            captured["hpg_v"] = np.asarray(-dp_dy / rho_0)
        return result

    pe_module._bc_ke_and_pressure_gradients = spy_gradients
    try:
        _, diagnostics = model.tendencies_with_diagnostics(
            state, surface_forcing=centred_sf, dt=DT,
        )
    finally:
        pe_module._bc_ke_and_pressure_gradients = real_gradients
    real_baro = ocean_model_module.barotropic_substeps_latlon_cgrid
    real_cor = barotropic_module.barotropic_coriolis_een_pre_step
    real_stress = pe_module.surface_stress_faces

    def spy_baro(state_mid, dt_s, n_substeps, grid, zc, cfg, **kwargs):
        result = real_baro(state_mid, dt_s, n_substeps, grid, zc, cfg, **kwargs)
        if kwargs.get("eta_init") is not None and "F_slow_u" not in captured:
            captured["F_slow_u"] = np.asarray(kwargs["F_slow_u"])
            captured["F_slow_v"] = np.asarray(kwargs["F_slow_v"])
        return result

    def spy_cor(*args, **kwargs):
        cor_u, cor_v = real_cor(*args, **kwargs)
        if "cor_u" not in captured:
            captured["cor_u"] = np.asarray(cor_u)
            captured["cor_v"] = np.asarray(cor_v)
        return cor_u, cor_v

    def spy_stress(surface_forcing, u_dtype, z_coord_arg, J, grid):
        result = real_stress(surface_forcing, u_dtype, z_coord_arg, J, grid)
        if result is not None and "tau_i_u" not in captured:
            tau_i_u, tau_j_v, dz_0_u, dz_0_v = result
            captured["tau_i_u"] = np.asarray(tau_i_u)
            captured["tau_j_v"] = np.asarray(tau_j_v)
            captured["dz_0_u"] = np.asarray(dz_0_u)
            captured["dz_0_v"] = np.asarray(dz_0_v)
        return result

    ocean_model_module.barotropic_substeps_latlon_cgrid = spy_baro
    barotropic_module.barotropic_coriolis_een_pre_step = spy_cor
    pe_module.surface_stress_faces = spy_stress
    try:
        with jax.disable_jit():
            model.step(state, DT, surface_forcing=sf)
    finally:
        ocean_model_module.barotropic_substeps_latlon_cgrid = real_baro
        barotropic_module.barotropic_coriolis_een_pre_step = real_cor
        pe_module.surface_stress_faces = real_stress
    required = {
        "F_slow_u", "F_slow_v", "cor_u", "cor_v",
        "tau_i_u", "tau_j_v", "dz_0_u", "dz_0_v",
        "keg_u", "keg_v", "hpg_u", "hpg_v",
    }
    if not required <= captured.keys():
        raise SystemExit(f"arm missed production hooks: {required - captured.keys()}")
    captured["diagnostics"] = diagnostics
    return captured


def _identity_except_stress(legacy, faithful) -> dict[str, Any]:
    exact = True
    checked = []
    for name in legacy._fields:
        if name in ("tau_x_prev", "tau_y_prev"):
            continue
        left = getattr(legacy, name)
        right = getattr(faithful, name)
        leaves_left = jax.tree_util.tree_leaves(left)
        leaves_right = jax.tree_util.tree_leaves(right)
        same = len(leaves_left) == len(leaves_right) and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(leaves_left, leaves_right)
        )
        exact &= same
        checked.append({"field": name, "exact": bool(same)})
    return {"all_nonstress_exact": bool(exact), "fields": checked}


def _zero_reference_gate(model_values, oracle_values) -> str:
    if np.count_nonzero(oracle_values) != 0:
        return "NOT_ZERO_REFERENCE"
    return (
        "UNMEASURED_ZERO_REFERENCE"
        if np.count_nonzero(model_values) == 0
        else "DEBT_NONZERO_AGAINST_ZERO_REFERENCE"
    )


def _attribute_error(term_error, residual) -> dict[str, Any]:
    term_error = np.asarray(term_error)
    residual = np.asarray(residual)
    residual_rms = _rms(residual)
    if residual_rms <= 0.0:
        raise RuntimeError("zero assembled residual has no attribution score")
    term_rms = _rms(term_error)
    corr = (
        float(np.corrcoef(term_error, residual)[0, 1])
        if term_error.size > 1 and np.std(term_error) > 0.0 else None
    )
    gain = term_rms / residual_rms
    removal = 1.0 - _rms(residual - term_error) / residual_rms
    if corr is not None and corr >= 0.99 and 0.90 <= gain <= 1.10 and removal >= 0.90:
        verdict = "CONFIRMS_CARRY"
    elif corr is not None and abs(corr) <= 0.20 and removal <= 0.10:
        verdict = "REFUTES_CARRY"
    else:
        verdict = "UNRESOLVED"
    return {
        "correlation_with_total_residual": corr,
        "rms_gain": gain,
        "residual_removal": removal,
        "verdict": verdict,
    }


def _attribution_controls() -> dict[str, Any]:
    phase = 2.0 * np.pi * np.arange(1024, dtype=np.float64) / 1024.0
    residual = np.sin(phase)
    confirming = _attribute_error(residual.copy(), residual)
    refuting = _attribute_error(np.cos(phase), residual)
    controls = {
        "identity_confirms": confirming["verdict"] == "CONFIRMS_CARRY",
        "orthogonal_refutes": refuting["verdict"] == "REFUTES_CARRY",
        "confirming_score": confirming,
        "refuting_score": refuting,
    }
    if not (controls["identity_confirms"] and controls["orthogonal_refutes"]):
        raise RuntimeError("attribution classifier controls failed")
    return controls


def _term_metric(lego, nemo, mask, expected_n, total_residual):
    lego_array = np.asarray(lego)
    nemo_array = np.asarray(nemo)
    if lego_array.shape != nemo_array.shape or lego_array.shape != mask.shape:
        raise RuntimeError("term metric shape changed")
    if not np.all(np.isfinite(lego_array)) or not np.all(np.isfinite(nemo_array)):
        raise RuntimeError("non-finite term metric operand")
    lego_values = lego_array[mask]
    nemo_values = nemo_array[mask]
    if lego_values.size != expected_n or nemo_values.size != expected_n:
        raise RuntimeError("term metric population changed")
    nemo_rms = _rms(nemo_values)
    if nemo_rms == 0.0:
        # A zero oracle term has no normalized-error statistic.  Preserve it as
        # an explicit structural-zero row and still test whether the legoESM
        # term error can carry the assembled residual.
        lego_rms = _rms(lego_values)
        zero_control = np.zeros_like(lego_values)
        planted = zero_control.copy()
        planted[0] = max(
            1.0e-30, 1.0e-12 * _rms(np.asarray(total_residual)[mask])
        )
        zero_gate = _zero_reference_gate(lego_values, nemo_values)
        control_gate = _zero_reference_gate(zero_control, nemo_values)
        planted_gate = _zero_reference_gate(planted, nemo_values)
        planted_fires = bool(
            control_gate == "UNMEASURED_ZERO_REFERENCE"
            and planted_gate == "DEBT_NONZERO_AGAINST_ZERO_REFERENCE"
            and not np.array_equal(zero_control, planted)
        )
        if not planted_fires:
            raise RuntimeError("zero-reference planted classifier control failed")
        metric = {
            "sample_count": int(lego_values.size),
            "reference_rms": 0.0,
            "model_rms": lego_rms,
            "max_abs_model": float(np.max(np.abs(lego_values))),
            "normalized_rms_error": None,
            "correlation": None,
            "campaign_gate": zero_gate,
            "structural_zero_reference": True,
            "zero_control": {
                "oracle_exact_zero": bool(np.count_nonzero(nemo_values) == 0),
                "model_exact_zero": bool(np.count_nonzero(lego_values) == 0),
                "identical_zero_gate": control_gate,
                "planted_nonzero_gate": planted_gate,
                "planted_nonzero_fires": planted_fires,
            },
        }
    else:
        metric = round1._metric(lego, nemo, mask, expected_n)
        metric["campaign_gate"] = round1._classify_metric(metric, None)
        metric["structural_zero_reference"] = False
    term_error = lego_values - nemo_values
    residual = np.asarray(total_residual)[mask]
    metric["attribution"] = _attribute_error(term_error, residual)
    return metric


def _counterfactual_score(legacy, faithful, nemo, mask):
    residual = (np.asarray(legacy) - np.asarray(nemo))[mask]
    prediction = (np.asarray(faithful) - np.asarray(legacy))[mask]
    corrected = (np.asarray(faithful) - np.asarray(nemo))[mask]
    rms_residual = _rms(residual)
    prediction_error = _rms(prediction + residual) / rms_residual
    corr = (
        float(np.corrcoef(prediction, -residual)[0, 1])
        if prediction.size > 1
        and np.std(prediction) > 0.0
        and np.std(residual) > 0.0
        else None
    )
    reduction = 1.0 - _rms(corrected) / rms_residual
    if corr is not None and prediction_error <= 0.10 and corr >= 0.99 and reduction >= 0.90:
        verdict = "CONFIRMS_BRIDGE_WIND_SOURCE"
    elif corr is not None and prediction_error >= 0.90 and corr <= 0.20 and reduction <= 0.10:
        verdict = "REFUTES_BRIDGE_WIND_SOURCE"
    else:
        verdict = "UNRESOLVED_BRIDGE_WIND_SOURCE"
    return {
        "prediction_normalized_error": prediction_error,
        "prediction_correlation": corr,
        "corrected_residual_reduction": reduction,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if os.environ.get("DINO_1226_LANE") != EXPECTED_LANE:
        raise SystemExit("DINO_1226_LANE=d180 is required")
    if os.environ.get("LEGOESM_NEMO_E3T") != "both":
        raise SystemExit("LEGOESM_NEMO_E3T=both is required")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX x64 are required")

    repo_root = Path(__file__).resolve().parents[4]
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty_before = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if dirty_before:
        raise SystemExit("clean worktree required:\n" + dirty_before)

    run_dir = Path(inherited.RUN_DIR).resolve()
    restart_path = run_dir / inherited.RESTART_FILE
    oracle_flags = _oracle_echo_flags(
        run_dir / "ocean.output",
        ("ln_apr_dyn", "ln_bt_fw", "ln_isfcav", "ln_drgice_imp"),
    )
    if any(value != "F" for value in oracle_flags.values()):
        raise SystemExit(f"registered dynspg_ts arm changed: {oracle_flags}")
    jpi, jpj, jpk, hls, _, _ = inherited._read_dims(str(run_dir))
    kt = int(inherited.dump_lane.KT_DUMP)
    dump_names = _register_dumps(kt)
    nemo = _load_terms(run_dir, jpi, jpj, jpk, hls, kt)

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0,
        lon_east_deg=49.0,
        sill_lon_m_deg=1.0,
    )
    grid = read_nemo_mesh_mask(str(run_dir / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(str(restart_path), nn_hls=0)
    bridge = bridge_nemo_to_legoesm_topo(
        grid,
        restart,
        periodic_i=True,
        full_step=True,
        omega=cfg.omega,
        carry_native_lat_deg=(
            cfg.tke_htau_evaluation == "nemo_literal"
            or cfg.gm_treguier_final_evaluation == "nemo_literal"
        ),
    )
    before = read_nemo_restart_before(str(restart_path), nn_hls=0)
    legacy = bridge_before_state_topo(
        bridge._replace(state=bridge.state), grid, before, periodic_i=True
    )
    model_config, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    if model_config.surface_stress_implicit:
        raise SystemExit("registered explicit surface-stress path changed")
    require_fp64(
        bridge.geometry, bridge.z_coord, legacy, context="split chain round2"
    )
    forcing = dino_lat_lon_surface_forcing_arrays(bridge.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)
    if sf is None or sf.tau_x is None or legacy.tau_x_prev is None:
        raise SystemExit("centred DINO stress operands missing")
    faithful = legacy._replace(
        tau_x_prev=jnp.asarray(sf.tau_x), tau_y_prev=jnp.asarray(sf.tau_y)
    )
    identity = _identity_except_stress(legacy, faithful)
    if not identity["all_nonstress_exact"]:
        raise SystemExit("faithful arm changed non-stress entry fields")

    h_k = compute_layer_thickness(
        legacy.eta.data,
        legacy.H_bathy.data,
        bridge.z_coord,
        min_water_column_m=model_config.min_water_column_m,
    )
    h_u = np.asarray(min_cell_to_uface(h_k))
    h_v = np.asarray(min_cell_to_vface(h_k, bridge.geometry))
    legacy_arm = _run_arm(
        legacy, sf, bridge.geometry, bridge.z_coord, model_config
    )
    faithful_arm = _run_arm(
        faithful, sf, bridge.geometry, bridge.z_coord, model_config
    )

    # NEMO momentum dumps stop at jpkm1; mesh_mask retains the terminal jpk
    # zero-thickness level.  Slice explicitly so the shape gate compares the
    # active momentum population rather than relying on that level being dry.
    umask3 = np.asarray(grid.umask[..., :jpk - 1], dtype=bool)
    vmask3 = np.asarray(grid.vmask[..., :jpk - 1], dtype=bool)
    umask2 = umask3[..., 0]
    vmask2 = vmask3[..., 0]
    if int(umask2.sum()) != EXPECTED_U or int(vmask2.sum()) != EXPECTED_V:
        raise SystemExit("registered U/V population changed")
    e3u = np.asarray(grid.e3u_0)
    e3v = np.asarray(grid.e3v_0)
    hu = np.asarray(grid.hu_0)
    hv = np.asarray(grid.hv_0)

    base_parts_u = sum(
        nemo["u"][name]
        for name in ("keg", "vertical_advection", "vorticity", "lateral_friction", "hpg")
    )
    base_parts_v = sum(
        nemo["v"][name]
        for name in ("keg", "vertical_advection", "vorticity", "lateral_friction", "hpg")
    )
    base_control_u = round1._metric(
        base_parts_u, nemo["u"]["stage06"], umask3, int(umask3.sum())
    )
    base_control_v = round1._metric(
        base_parts_v, nemo["v"]["stage06"], vmask3, int(vmask3.sum())
    )
    if (
        base_control_u["normalized_rms_error"] > BASE_RECONSTRUCTION_BAR
        or base_control_v["normalized_rms_error"] > BASE_RECONSTRUCTION_BAR
    ):
        raise SystemExit(f"NEMO base reconstruction failed: {base_control_u} {base_control_v}")
    planted_base_u = base_parts_u.copy()
    planted_base_u[umask3] += 1.0e-6 * _rms(nemo["u"]["stage06"][umask3])
    planted_base_metric = round1._metric(
        planted_base_u, nemo["u"]["stage06"], umask3, int(umask3.sum())
    )
    base_reconstruction_control = {
        "bar": BASE_RECONSTRUCTION_BAR,
        "u_pass": base_control_u["normalized_rms_error"] <= BASE_RECONSTRUCTION_BAR,
        "v_pass": base_control_v["normalized_rms_error"] <= BASE_RECONSTRUCTION_BAR,
        "planted_u_normalized_error": planted_base_metric["normalized_rms_error"],
        "planted_u_fires": (
            planted_base_metric["normalized_rms_error"] > BASE_RECONSTRUCTION_BAR
        ),
    }
    if not base_reconstruction_control["planted_u_fires"]:
        raise SystemExit("NEMO base reconstruction plant did not breach bar")

    base_u = _nemo_depth_mean(base_parts_u, e3u, umask3, hu)
    base_v = _nemo_depth_mean(base_parts_v, e3v, vmask3, hv)
    nemo_cor_u = base_u + nemo["u"]["drag"] + nemo["u"]["wind"] - nemo["u"]["final"]
    nemo_cor_v = base_v + nemo["v"]["drag"] + nemo["v"]["wind"] - nemo["v"]["final"]
    ledger_u = base_u - nemo_cor_u + nemo["u"]["drag"] + nemo["u"]["wind"]
    ledger_v = base_v - nemo_cor_v + nemo["v"]["drag"] + nemo["v"]["wind"]
    ledger_control_u = round1._metric(
        ledger_u, nemo["u"]["final"], umask2, EXPECTED_U
    )
    ledger_control_v = round1._metric(
        ledger_v, nemo["v"]["final"], vmask2, EXPECTED_V
    )
    # The inferred term subtracts the final field and then reconstructs it;
    # that inverse operation is algebraically exact but not bit-invertible.
    # Bind acceptance to an explicit fp64 operation-count roundoff envelope,
    # not to a relaxed campaign physics bar.
    eps = np.finfo(np.float64).eps
    ledger_diff_u = np.abs(ledger_u - nemo["u"]["final"])
    ledger_diff_v = np.abs(ledger_v - nemo["v"]["final"])
    ledger_bound_u = 8.0 * eps * (
        np.abs(base_u) + np.abs(nemo_cor_u) + np.abs(nemo["u"]["drag"])
        + np.abs(nemo["u"]["wind"]) + np.abs(nemo["u"]["final"])
    )
    ledger_bound_v = 8.0 * eps * (
        np.abs(base_v) + np.abs(nemo_cor_v) + np.abs(nemo["v"]["drag"])
        + np.abs(nemo["v"]["wind"]) + np.abs(nemo["v"]["final"])
    )
    ledger_roundoff_u = bool(np.all(ledger_diff_u[umask2] <= ledger_bound_u[umask2]))
    ledger_roundoff_v = bool(np.all(ledger_diff_v[vmask2] <= ledger_bound_v[vmask2]))
    # A material perturbation must breach the same bound.
    ledger_plant_u = ledger_u.copy()
    ledger_plant_u[umask2] += 1.0e-12 * _rms(nemo["u"]["final"][umask2])
    ledger_plant_fires = bool(
        np.any(np.abs(ledger_plant_u - nemo["u"]["final"])[umask2]
               > ledger_bound_u[umask2])
    )
    if not (ledger_roundoff_u and ledger_roundoff_v and ledger_plant_fires):
        raise SystemExit("inferred-Coriolis ledger exceeds fp64 roundoff envelope")

    drag_u, drag_v = _drag_contribution(
        legacy, h_k, jnp.asarray(h_u), jnp.asarray(h_v),
        bridge.z_coord, model_config, bridge.geometry,
    )

    def lego_terms(arm, component):
        diagnostics = arm["diagnostics"]
        if component == "u":
            mapper, thickness, mask = _u_to_nemo, h_u, np.asarray(legacy.u_mask.data)
            fields = {
                "kinetic_energy_gradient": arm["keg_u"],
                "vertical_advection": diagnostics.vertadv_u.data,
                "vorticity": diagnostics.vortcor_u.data,
                "lateral_friction": (
                    diagnostics.Ah_lap_u.data + diagnostics.Bh_bilap_u.data
                    + diagnostics.Cs_smag_u.data + diagnostics.Cl_leith_u.data
                ),
                "pressure_gradient": arm["hpg_u"],
                "vertical_friction": diagnostics.Av_vert_u.data,
            }
            direct_drag, cor = drag_u, arm["cor_u"]
            tau, dz0 = arm["tau_i_u"], arm["dz_0_u"]
        else:
            mapper, thickness, mask = _v_to_nemo, h_v, np.asarray(legacy.v_mask.data)
            fields = {
                "kinetic_energy_gradient": arm["keg_v"],
                "vertical_advection": diagnostics.vertadv_v.data,
                "vorticity": diagnostics.vortcor_v.data,
                "lateral_friction": (
                    diagnostics.Ah_lap_v.data + diagnostics.Bh_bilap_v.data
                    + diagnostics.Cs_smag_v.data + diagnostics.Cl_leith_v.data
                ),
                "pressure_gradient": arm["hpg_v"],
                "vertical_friction": diagnostics.Av_vert_v.data,
            }
            direct_drag, cor = drag_v, arm["cor_v"]
            tau, dz0 = arm["tau_j_v"], arm["dz_0_v"]
        out = {
            name: mapper(_live_depth_mean(np.asarray(field), thickness, mask))
            for name, field in fields.items()
        }
        out["cor_removal"] = -mapper(np.asarray(cor))
        out["drag"] = mapper(direct_drag)
        # Reuse zu_frc_write_ledger.py's production-spy reconstruction:
        # external stress is a top-cell kick tau/(rho0*dz0), then the exact
        # F_slow thickness reduction weights it by h[...,0]/sum(h).
        wind0 = tau / (
            float(model_config.constants.rho_0) * np.maximum(dz0, 1.0e-10)
        )
        out["wind"] = mapper(
            wind0 * thickness[..., 0]
            / np.maximum(np.sum(thickness, axis=-1), 1.0e-10) * mask
        )
        return out

    legacy_terms_u = lego_terms(legacy_arm, "u")
    legacy_terms_v = lego_terms(legacy_arm, "v")
    faithful_terms_u = lego_terms(faithful_arm, "u")
    faithful_terms_v = lego_terms(faithful_arm, "v")
    nemo_terms_u = {
        "kinetic_energy_gradient": _nemo_depth_mean(
            nemo["u"]["keg"], e3u, umask3, hu
        ),
        "vertical_advection": _nemo_depth_mean(
            nemo["u"]["vertical_advection"], e3u, umask3, hu
        ),
        "vorticity": _nemo_depth_mean(nemo["u"]["vorticity"], e3u, umask3, hu),
        "lateral_friction": _nemo_depth_mean(
            nemo["u"]["lateral_friction"], e3u, umask3, hu
        ),
        "pressure_gradient": _nemo_depth_mean(
            nemo["u"]["hpg"], e3u, umask3, hu
        ),
        "cor_removal": -nemo_cor_u,
        "drag": nemo["u"]["drag"],
        "wind": nemo["u"]["wind"],
    }
    nemo_terms_v = {
        "kinetic_energy_gradient": _nemo_depth_mean(
            nemo["v"]["keg"], e3v, vmask3, hv
        ),
        "vertical_advection": _nemo_depth_mean(
            nemo["v"]["vertical_advection"], e3v, vmask3, hv
        ),
        "vorticity": _nemo_depth_mean(nemo["v"]["vorticity"], e3v, vmask3, hv),
        "lateral_friction": _nemo_depth_mean(
            nemo["v"]["lateral_friction"], e3v, vmask3, hv
        ),
        "pressure_gradient": _nemo_depth_mean(
            nemo["v"]["hpg"], e3v, vmask3, hv
        ),
        "cor_removal": -nemo_cor_v,
        "drag": nemo["v"]["drag"],
        "wind": nemo["v"]["wind"],
    }

    F_legacy_u = _u_to_nemo(legacy_arm["F_slow_u"])
    F_legacy_v = _v_to_nemo(legacy_arm["F_slow_v"])
    F_faithful_u = _u_to_nemo(faithful_arm["F_slow_u"])
    F_faithful_v = _v_to_nemo(faithful_arm["F_slow_v"])
    residual_u = F_legacy_u - nemo["u"]["final"]
    residual_v = F_legacy_v - nemo["v"]["final"]
    faithful_residual_u = F_faithful_u - nemo["u"]["final"]
    faithful_residual_v = F_faithful_v - nemo["v"]["final"]

    total_metrics = {
        "legacy_u": round1._metric(F_legacy_u, nemo["u"]["final"], umask2, EXPECTED_U),
        "legacy_v": round1._metric(F_legacy_v, nemo["v"]["final"], vmask2, EXPECTED_V),
        "faithful_u": round1._metric(F_faithful_u, nemo["u"]["final"], umask2, EXPECTED_U),
        "faithful_v": round1._metric(F_faithful_v, nemo["v"]["final"], vmask2, EXPECTED_V),
    }
    for metric in total_metrics.values():
        metric["campaign_gate"] = round1._classify_metric(metric, None)

    term_metrics: dict[str, dict[str, Any]] = {"u": {}, "v": {}}
    ordered = (
        "kinetic_energy_gradient", "vertical_advection", "vorticity",
        "lateral_friction", "pressure_gradient", "cor_removal", "drag",
        "wind",
    )
    for name in ordered:
        term_metrics["u"][name] = _term_metric(
            legacy_terms_u[name], nemo_terms_u[name], umask2, EXPECTED_U, residual_u
        )
        term_metrics["v"][name] = _term_metric(
            legacy_terms_v[name], nemo_terms_v[name], vmask2, EXPECTED_V, residual_v
        )
    term_metrics["u"]["wind_faithful"] = _term_metric(
        faithful_terms_u["wind"], nemo_terms_u["wind"], umask2, EXPECTED_U, residual_u
    )
    term_metrics["v"]["wind_faithful"] = _term_metric(
        faithful_terms_v["wind"], nemo_terms_v["wind"], vmask2, EXPECTED_V, residual_v
    )
    faithful_term_metrics: dict[str, dict[str, Any]] = {"u": {}, "v": {}}
    for name in ordered:
        faithful_term_metrics["u"][name] = _term_metric(
            faithful_terms_u[name], nemo_terms_u[name], umask2, EXPECTED_U,
            faithful_residual_u,
        )
        faithful_term_metrics["v"][name] = _term_metric(
            faithful_terms_v[name], nemo_terms_v[name], vmask2, EXPECTED_V,
            faithful_residual_v,
        )

    counterfactual = {
        "u": _counterfactual_score(
            F_legacy_u, F_faithful_u, nemo["u"]["final"], umask2
        ),
        "v": _counterfactual_score(
            F_legacy_v, F_faithful_v, nemo["v"]["final"], vmask2
        ),
    }

    prediction_u = F_faithful_u - F_legacy_u
    receiver = tuple(
        int(value) for value in np.unravel_index(
            np.argmax(np.where(umask2, np.abs(prediction_u), -np.inf)),
            prediction_u.shape,
        )
    )
    donor = (receiver[0], (receiver[1] + 1) % prediction_u.shape[1])
    equalized_tau = np.asarray(legacy.tau_x_prev).copy()
    equalized_tau[donor] = equalized_tau[receiver]
    equalized = legacy._replace(tau_x_prev=jnp.asarray(equalized_tau))
    equalized_arm = _run_arm(
        equalized, sf, bridge.geometry, bridge.z_coord, model_config
    )
    F_equalized_u = _u_to_nemo(equalized_arm["F_slow_u"])
    equalized_score = _counterfactual_score(
        F_legacy_u, F_equalized_u, nemo["u"]["final"], umask2
    )
    donor_control = {
        "receiver": list(receiver),
        "east_donor": list(donor),
        "faithful_score": counterfactual["u"],
        "equalized_score": equalized_score,
        "changes_verdict_path": (
            equalized_score["verdict"] != counterfactual["u"]["verdict"]
        ),
    }
    if not donor_control["changes_verdict_path"]:
        raise SystemExit("east-donor equalization control did not change owner verdict")

    classifier_controls = round1._controls(
        (F_legacy_u, nemo["u"]["final"], umask2), EXPECTED_U
    )
    if not all(
        classifier_controls[key]
        for key in (
            "identical_array_zero",
            "planted_identity_flips_campaign_gate",
            "actual_binding_perturbation_changes_score",
            "zero_shift_is_best",
        )
    ):
        raise SystemExit("campaign classifier/alignment controls failed")

    term_sum_error_u = sum(
        legacy_terms_u[name] - nemo_terms_u[name] for name in ordered
    )
    term_sum_error_v = sum(
        legacy_terms_v[name] - nemo_terms_v[name] for name in ordered
    )
    closure_u = _rms((term_sum_error_u - residual_u)[umask2]) / _rms(residual_u[umask2])
    closure_v = _rms((term_sum_error_v - residual_v)[vmask2]) / _rms(residual_v[vmask2])
    planted_term_sum_u = term_sum_error_u.copy()
    planted_term_sum_u[umask2] += 1.0e-6 * _rms(residual_u[umask2])
    planted_closure_u = _rms(
        (planted_term_sum_u - residual_u)[umask2]
    ) / _rms(residual_u[umask2])
    closure_controls = {
        "bar": TERM_CLOSURE_BAR,
        "u_pass": closure_u <= TERM_CLOSURE_BAR,
        "v_pass": closure_v <= TERM_CLOSURE_BAR,
        "planted_u_normalized_closure": planted_closure_u,
        "planted_u_fires": planted_closure_u > TERM_CLOSURE_BAR,
    }
    if not (
        closure_controls["u_pass"]
        and closure_controls["v_pass"]
        and closure_controls["planted_u_fires"]
    ):
        raise SystemExit(f"signed term-error closure failed: {closure_controls}")
    attribution_controls = _attribution_controls()
    cancellation = {"u": {}, "v": {}}
    for name in ordered:
        for component, residual, leading, mask in (
            ("u", residual_u, legacy_terms_u[name] - nemo_terms_u[name], umask2),
            ("v", residual_v, legacy_terms_v[name] - nemo_terms_v[name], vmask2),
        ):
            pair = leading + (
                (legacy_terms_u["wind"] - nemo_terms_u["wind"])
                if component == "u" and name != "wind"
                else (legacy_terms_v["wind"] - nemo_terms_v["wind"])
                if component == "v" and name != "wind"
                else 0.0
            )
            r = residual[mask]
            p = np.asarray(pair)[mask]
            cancellation[component][f"{name}+wind"] = {
                "residual_removal": 1.0 - _rms(r - p) / _rms(r)
            }

    clock = _restart_clock(restart_path)
    script_dir = Path(__file__).resolve().parent
    oracle = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    prereg = repo_root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round2.md"
    provenance_paths = {
        "probe": Path(__file__).resolve(),
        "round1_wrapper": script_dir / "split_explicit_momentum_chain_round1.py",
        "inherited_probe": script_dir / "spg_substep_chain.py",
        "preregistration": prereg,
        "fidelity_bar_gate": Path(bar_gate.__file__).resolve(),
        "production_ocean_model": Path(ocean_model_module.__file__).resolve(),
        "production_barotropic": Path(barotropic_module.__file__).resolve(),
        "production_pe": Path(pe_module.__file__).resolve(),
        "production_bridge": Path(bridge_module.__file__).resolve(),
        "production_dino_card": Path(dino_module.__file__).resolve(),
        "nemo_stpmlf": oracle / "cfgs/DINO/MY_SRC/stpmlf.F90",
        "nemo_dynadv": oracle / "cfgs/DINO/MY_SRC/dynadv.F90",
        "nemo_dynvor": oracle / "cfgs/DINO/MY_SRC/dynvor.F90",
        "nemo_dynldf": oracle / "cfgs/DINO/MY_SRC/dynldf.F90",
        "nemo_dynhpg": oracle / "cfgs/DINO/MY_SRC/dynhpg.F90",
        "nemo_dynspg_ts": oracle / "cfgs/DINO/MY_SRC/dynspg_ts.F90",
        "nemo_dynzdf": oracle / "cfgs/DINO/MY_SRC/dynzdf.F90",
    }
    inputs = (
        "mesh_mask.nc", inherited.RESTART_FILE, "ocean.output", "namelist_cfg",
        "namelist_ref", "output.namelist.dyn", "nemo",
    )
    dirty_after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if dirty_after:
        raise SystemExit("worktree changed during measurement")

    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round2-v1",
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "git": {"commit": git_sha, "clean_before": True, "clean_after": True},
        "lane": EXPECTED_LANE,
        "kt": kt,
        "run_dir": str(run_dir),
        "restart_clock": clock,
        "oracle_runtime_flags": oracle_flags,
        "lateral_friction_levels": {
            "lego_live_barotropic_first_pass": "NOW",
            "lego_later_3d_dissipative_pass": "BEFORE_BAROTROPIC_RESULT_DISCARDED",
            "nemo_dynldf_operand": "BEFORE",
            "lego_source": "ocean_model_latlon_cgrid.py:8950-9030",
            "nemo_source": "stpmlf.F90:319-322; dynldf.F90:69-119",
        },
        "runtime": {
            "backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "e3t_mode": os.environ.get("LEGOESM_NEMO_E3T"),
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "cpu_device": str(jax.devices("cpu")[0]),
        },
        "effective_config_repr_sha256": hashlib.sha256(repr(cfg).encode()).hexdigest(),
        "derived_model_config_repr_sha256": hashlib.sha256(
            repr(model_config).encode()
        ).hexdigest(),
        "nemo_base_reconstruction": {"u": base_control_u, "v": base_control_v},
        "inferred_coriolis_ledger": {
            "u": ledger_control_u,
            "v": ledger_control_v,
            "fp64_roundoff_envelope_pass_u": ledger_roundoff_u,
            "fp64_roundoff_envelope_pass_v": ledger_roundoff_v,
            "planted_material_offset_fires": ledger_plant_fires,
            "max_abs_difference_u": float(np.max(ledger_diff_u[umask2])),
            "max_abs_difference_v": float(np.max(ledger_diff_v[vmask2])),
            "max_roundoff_bound_u": float(np.max(ledger_bound_u[umask2])),
            "max_roundoff_bound_v": float(np.max(ledger_bound_v[vmask2])),
        },
        "vertical_friction": {
            "status": "STRUCTURAL_ZERO_NOT_AN_OPERAND",
            "nemo_source": "stpmlf.F90:332 before :396; dynzdf.F90:134-337",
            "legacy_u_max_abs": float(np.max(np.abs(legacy_terms_u["vertical_friction"]))),
            "legacy_v_max_abs": float(np.max(np.abs(legacy_terms_v["vertical_friction"]))),
        },
        "total_metrics": total_metrics,
        "term_metrics": term_metrics,
        "faithful_term_metrics": faithful_term_metrics,
        "counterfactual": counterfactual,
        "term_error_sum_closure": {
            "u_normalized_to_total_residual": closure_u,
            "v_normalized_to_total_residual": closure_v,
        },
        "cancellation_pairs": cancellation,
        "controls": {
            "entry_identity": identity,
            "donor_equalization": donor_control,
            "campaign_classifier_and_alignment": classifier_controls,
            "base_reconstruction": base_reconstruction_control,
            "term_error_closure": closure_controls,
            "attribution_classifier": attribution_controls,
        },
        "ordered_continuation": {
            "row_1_1": "OPEN_SOURCE_LOCALIZED_FAITHFUL_TOTAL_DEBT",
            "row_1_2": "ORDERED_BLOCKED_UNLESS_FAITHFUL_TOTAL_AT_BAR",
            "row_1_3": "ORDERED_BLOCKED_UNLESS_FAITHFUL_TOTAL_AT_BAR",
            "rows_2_to_6": "ORDERED_BLOCKED",
        },
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance_paths.items()
        },
        "dump_sha256": {name: _sha256(run_dir / name) for name in dump_names},
        "run_input_sha256": {name: _sha256(run_dir / name) for name in inputs},
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    print("ROW 1.1 TERM PEEL")
    for component in ("u", "v"):
        legacy_metric = total_metrics[f"legacy_{component}"]
        faithful_metric = total_metrics[f"faithful_{component}"]
        print(
            f"  {component}: legacy E={legacy_metric['normalized_rms_error']:.9e} "
            f"{legacy_metric['campaign_gate']} -> faithful "
            f"E={faithful_metric['normalized_rms_error']:.9e} "
            f"{faithful_metric['campaign_gate']}  "
            f"counterfactual={counterfactual[component]['verdict']}"
        )
        for name in ordered:
            attribution = term_metrics[component][name]["attribution"]
            corr_value = attribution["correlation_with_total_residual"]
            corr_text = "null" if corr_value is None else f"{corr_value:+.6f}"
            print(
                f"    {name}: corrR={corr_text} "
                f"gain={attribution['rms_gain']:.6f} "
                f"removal={attribution['residual_removal']:.6f} "
                f"{attribution['verdict']}"
            )
        if component == "u":
            print("    post-faithful U tail:")
            for name in ordered:
                attribution = faithful_term_metrics["u"][name]["attribution"]
                corr_value = attribution["correlation_with_total_residual"]
                corr_text = "null" if corr_value is None else f"{corr_value:+.6f}"
                print(
                    f"      {name}: "
                    f"corrR={corr_text} "
                    f"gain={attribution['rms_gain']:.6f} "
                    f"removal={attribution['residual_removal']:.6f} "
                    f"{attribution['verdict']}"
                )
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
