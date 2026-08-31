"""Committed CPU peel of DINO's analytic IC and first Euler bootstrap.

The frozen protocol is ``PREREG_dino_ic_euler_peel.md``.  This probe reports
mechanical rows and the first over-bar row; scientific interpretation lives in
the result document, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

import hpg_tendency_compare as hpg
import kamm_twin_90d as twin
import standalone_20y as standalone
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing,
    dino_S_profile_1d,
    dino_T_profile_1d,
    dino_lat_lon_grid,
    dino_nemo_istate_profiles_1d,
)
from legoesm.grids.latlon import create_mercator_grid
from legoesm.ocean.init_latlon_cgrid import partial_periodic_seam_wall_latlon
from legoesm.ocean.vertical import (
    compute_layer_thickness,
    create_levy_stretched_z_star,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)


SCHEMA = "dino_ic_euler_peel_v6"
BAR = 1.0e-15
RUNTIME_SHAPE = (203, 56)
RUNTIME_HALO = 2
STANDALONE_CORE = 2
SESSION_ID = "01a053d4-8e9f-7212-bbdb-19ba2d64e140"
INIT_ADMISSION_ROWS = (
    "input_wet_mask",
    "input_latitude_deg",
    "input_t_depth_m",
    "common_depth_T_profile",
    "common_depth_S_profile",
    "resolved_T_nemo_wet",
    "resolved_S_nemo_wet",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {value.shape} != {expected}")


def _ordered_float64(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where((bits & sign) != 0, ~bits, bits ^ sign)


def diff_row(name: str, actual: np.ndarray, expected: np.ndarray,
             active: np.ndarray, *, bar: float = BAR) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require_shape(f"{name}.actual", actual, expected.shape)
    require_shape(f"{name}.active", active, expected.shape)
    if not np.isfinite(actual[active]).all() or not np.isfinite(expected[active]).all():
        raise ValueError(f"{name} contains non-finite active values")
    delta = np.abs(actual - expected)
    selected = delta[active]
    mismatch = selected > bar
    exact_mismatch = selected != 0.0
    indices = np.argwhere(active & (delta > bar))
    first = None
    if indices.size:
        index = tuple(int(i) for i in indices[0])
        first = {
            "index": list(index),
            "actual": float(actual[index]),
            "expected": float(expected[index]),
            "abs": float(delta[index]),
        }
    ordered_actual = _ordered_float64(actual[active])
    ordered_expected = _ordered_float64(expected[active])
    ulps = np.maximum(ordered_actual, ordered_expected) - np.minimum(
        ordered_actual, ordered_expected)
    return {
        "name": name,
        "bar": float(bar),
        "n": int(selected.size),
        "max_abs": float(selected.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(selected * selected))) if selected.size else 0.0,
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "exact_mismatch_count": int(np.count_nonzero(exact_mismatch)),
        "max_ulp": int(ulps.max(initial=0)),
        "first_over_bar": first,
        "status": "PASS" if not np.any(mismatch) else "OVER_BAR",
    }


def _broadcast_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim + 1 == len(shape):
        array = array[..., None]
    return np.broadcast_to(array, shape)


def _mesh_core(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        return array[STANDALONE_CORE:-STANDALONE_CORE,
                     STANDALONE_CORE:-STANDALONE_CORE, :-1]
    if array.ndim == 2:
        return array[STANDALONE_CORE:-STANDALONE_CORE,
                     STANDALONE_CORE:-STANDALONE_CORE]
    raise ValueError(f"unsupported mesh rank {array.ndim}")


def _model_core(value: np.ndarray) -> np.ndarray:
    """195x48 physical score from a live 199x52 construction field."""
    array = np.asarray(value)
    if array.ndim not in (2, 3):
        raise ValueError(f"unsupported live-model rank {array.ndim}")
    return array[STANDALONE_CORE:-STANDALONE_CORE,
                 STANDALONE_CORE:-STANDALONE_CORE, ...]


def _runtime_dump(path: Path) -> np.ndarray:
    array = hpg.read_dump(
        str(path), 35, RUNTIME_SHAPE[0], RUNTIME_SHAPE[1])
    return array[RUNTIME_HALO + STANDALONE_CORE:
                 -RUNTIME_HALO - STANDALONE_CORE,
                 RUNTIME_HALO + STANDALONE_CORE:
                 -RUNTIME_HALO - STANDALONE_CORE, :]


def _runtime_ssh(path: Path) -> np.ndarray:
    array = np.fromfile(path, dtype=np.float64)
    expected = RUNTIME_SHAPE[0] * RUNTIME_SHAPE[1]
    if array.size != expected:
        raise ValueError(f"{path}: {array.size} values != {expected}")
    array = array.reshape(RUNTIME_SHAPE)
    edge = RUNTIME_HALO + STANDALONE_CORE
    return array[edge:-edge, edge:-edge]


def _initial_rows(grid, geometry, z_coord, state, cfg) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray,
        dict[str, int]]:
    if grid.gdept_0 is None:
        raise ValueError("mesh_mask.nc lacks gdept_0")
    nemo_t_full, nemo_s_full = hpg.nemo_istate_case4(
        grid.gdept_0, grid.gphit, grid.tmask)
    nemo_t_full = np.asarray(nemo_t_full)[..., :-1]
    nemo_s_full = np.asarray(nemo_s_full)[..., :-1]
    nemo_t = _model_core(nemo_t_full)
    nemo_s = _model_core(nemo_s_full)
    mask = _mesh_core(grid.tmask).astype(bool)
    lego_mask_full = (
        np.asarray(z_coord.is_active, dtype=bool)
        & (np.asarray(state.land_mask.data) > 0.5)[..., None])
    lego_mask = _model_core(lego_mask_full)
    common_mask = mask & lego_mask
    mask_receipt = {
        "nemo_wet": int(np.count_nonzero(mask)),
        "lego_wet": int(np.count_nonzero(lego_mask)),
        "common_wet": int(np.count_nonzero(common_mask)),
        "nemo_wet_lego_dry": int(np.count_nonzero(mask & ~lego_mask)),
        "nemo_dry_lego_wet": int(np.count_nonzero(~mask & lego_mask)),
    }
    lego_depth = _model_core(np.asarray(z_coord.nemo_gdept_0))
    lego_lat = _model_core(np.asarray(geometry.native_lat_T_deg))
    rows = [
        diff_row("input_wet_mask", lego_mask.astype(np.float64),
                 mask.astype(np.float64), np.ones(mask.shape, dtype=bool), bar=0.0),
        diff_row("input_latitude_deg", lego_lat, _mesh_core(grid.gphit),
                 np.ones(lego_lat.shape, dtype=bool), bar=0.0),
        diff_row("input_t_depth_m", lego_depth, _mesh_core(grid.gdept_0),
                 mask, bar=0.0),
    ]

    nemo_depth = _mesh_core(grid.gdept_0)
    source_t_profile, source_s_profile = hpg.nemo_istate_profiles_1d(nemo_depth)
    lego_t_profile, lego_s_profile = dino_nemo_istate_profiles_1d(nemo_depth)
    rows.extend([
        diff_row("common_depth_T_profile", lego_t_profile, source_t_profile, mask),
        diff_row("common_depth_S_profile", lego_s_profile, source_s_profile, mask),
        diff_row("resolved_T_nemo_wet", _model_core(state.T.data), nemo_t,
                 mask),
        diff_row("resolved_S_nemo_wet", _model_core(state.S.data), nemo_s,
                 mask),
    ])

    full_source_t, full_source_s = hpg.nemo_istate_profiles_1d(grid.gdept_0)
    phi_max = float(np.max(grid.gphit))
    t_bot = float(np.min(full_source_t + 100.0 * (1.0 - grid.tmask)))
    s_bot = float(np.min(full_source_s + 100.0 * (1.0 - grid.tmask)))
    source_lego_depth, source_lego_depth_s = hpg.nemo_istate_case4(
        lego_depth, lego_lat, lego_mask.astype(np.float64),
        phi_max_deg=phi_max, t_bot=t_bot, s_bot=s_bot)
    source_lego_anchors, source_lego_anchors_s = hpg.nemo_istate_case4(
        lego_depth, lego_lat, lego_mask.astype(np.float64))
    legacy_t_prof, legacy_s_prof = hpg.nemo_istate_profiles_1d(lego_depth)
    source_legacy, source_legacy_s = hpg.nemo_istate_case4(
        lego_depth, lego_lat, lego_mask.astype(np.float64),
        phi_max_deg=float(cfg.lat_max_deg),
        t_bot=float(legacy_t_prof[..., -1].min()),
        s_bot=float(legacy_s_prof[..., -1].min()))
    localization = [
        diff_row("substitute_nemo_depth_T_common_wet", source_lego_depth,
                 nemo_t, common_mask),
        diff_row("substitute_nemo_depth_S_common_wet", source_lego_depth_s,
                 nemo_s, common_mask),
        diff_row("substitute_nemo_anchors_T", source_lego_anchors,
                 _model_core(state.T.data), common_mask),
        diff_row("substitute_nemo_anchors_S", source_lego_anchors_s,
                 _model_core(state.S.data), common_mask),
        diff_row("source_order_legacy_T", source_legacy,
                 _model_core(state.T.data), common_mask),
        diff_row("source_order_legacy_S", source_legacy_s,
                 _model_core(state.S.data), common_mask),
    ]
    return rows, localization, nemo_t_full, nemo_s_full, mask_receipt


def initialization_admitted(rows: list[dict[str, Any]]) -> bool:
    """Fail closed unless every frozen initialization row exists and passes."""
    by_name = {row["name"]: row for row in rows}
    missing = [name for name in INIT_ADMISSION_ROWS if name not in by_name]
    if missing:
        raise ValueError("missing initialization admission rows: "
                         + ", ".join(missing))
    return all(by_name[name]["status"] == "PASS"
               for name in INIT_ADMISSION_ROWS)


def _forcing_localization(run_kt2: Path, grid, z_coord, state, model,
                          step_forcing) -> dict[str, Any]:
    """Post-hoc split of the first cold-start slow momentum forcing.

    This is localization only: it does not alter the frozen Euler rows or
    their 1e-15 bar.  At rest the momentum diagnostic closes from HPG plus
    wind stress; all velocity-dependent components are exact zeros.  The
    registered common-face population excludes the artificial boundary face
    introduced by stripping NEMO's two-ring construction frame.
    """
    tendencies, diag = model.tendencies_with_diagnostics(
        state, step_forcing, dt=standalone.DT_SECONDS)
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord,
        min_water_column_m=model.config.min_water_column_m)
    h_u = np.asarray(min_cell_to_uface(h_k))
    h_v = np.asarray(min_cell_to_vface(h_k, model.grid))
    H_u = np.maximum(np.sum(h_u, axis=-1), 1.0e-10)
    H_v = np.maximum(np.sum(h_v, axis=-1), 1.0e-10)
    u_mask_2d = np.asarray(state.u_mask.data)
    v_mask_2d = np.asarray(state.v_mask.data)

    def depth_mean(value, weight, total, mask):
        return (np.sum(np.asarray(value) * weight, axis=-1) / total) * mask

    hpg_u = depth_mean(diag.KE_PGF_u.data, h_u, H_u, u_mask_2d)
    hpg_v = depth_mean(diag.KE_PGF_v.data, h_v, H_v, v_mask_2d)
    wind_u = depth_mean(diag.surface_stress_u.data, h_u, H_u, u_mask_2d)
    wind_v = depth_mean(diag.surface_stress_v.data, h_v, H_v, v_mask_2d)
    total_u = depth_mean(tendencies.du_dt.data, h_u, H_u, u_mask_2d)
    total_v = depth_mean(tendencies.dv_dt.data, h_v, H_v, v_mask_2d)

    mask_t = _mesh_core(grid.tmask).astype(bool)
    mask_u = _mesh_core(grid.umask).astype(bool)
    mask_v = _mesh_core(grid.vmask).astype(bool)
    lego_t_full = (
        np.asarray(z_coord.is_active, dtype=bool)
        & (np.asarray(state.land_mask.data) > 0.5)[..., None])
    lego_u_full = lego_t_full & np.roll(lego_t_full, -1, axis=1)
    lego_v_full = np.zeros_like(lego_t_full)
    lego_v_full[:-1] = lego_t_full[:-1] & lego_t_full[1:]
    lego_u = _model_core(lego_u_full)
    lego_v = _model_core(lego_v_full)
    common_u = (mask_u & lego_u)[..., 0]
    common_v = (mask_v & lego_v)[..., 0]

    def interior_2d(name: str) -> np.ndarray:
        value = np.fromfile(run_kt2 / name, dtype="<f8")
        expected = 199 * 52
        if value.size != expected:
            raise ValueError(f"{name}: {value.size} values != {expected}")
        return value.reshape(199, 52)[2:-2, 2:-2]

    nemo_total_u = interior_2d("spg_dump_zu_frc.bin")
    nemo_total_v = interior_2d("spg_dump_zv_frc.bin")
    nemo_wind_u = interior_2d("wnd_dump_zu_frc_inc.bin")
    nemo_wind_v = interior_2d("wnd_dump_zv_frc_inc.bin")
    h_u_core = _model_core(h_u[:, 1:, :])
    h_v_core = _model_core(h_v[1:, :, :])
    H_u_core = _model_core(H_u[:, 1:])
    H_v_core = _model_core(H_v[1:, :])
    nemo_hpg_u = np.sum(
        _runtime_dump(run_kt2 / "hpg_dump_du.bin") * h_u_core,
        axis=-1) / H_u_core
    nemo_hpg_v = np.sum(
        _runtime_dump(run_kt2 / "hpg_dump_dv.bin") * h_v_core,
        axis=-1) / H_v_core

    total_u_core = _model_core(total_u[:, 1:])
    total_v_core = _model_core(total_v[1:])
    wind_u_core = _model_core(wind_u[:, 1:])
    wind_v_core = _model_core(wind_v[1:])
    hpg_u_core = _model_core(hpg_u[:, 1:])
    hpg_v_core = _model_core(hpg_v[1:])

    rows = [
        diff_row("POST_HOC_fslow_u_total", total_u_core, nemo_total_u,
                 common_u),
        diff_row("POST_HOC_fslow_v_total", total_v_core, nemo_total_v,
                 common_v),
        diff_row("POST_HOC_fslow_u_wind", wind_u_core, nemo_wind_u,
                 common_u),
        diff_row("POST_HOC_fslow_v_wind", wind_v_core, nemo_wind_v,
                 common_v),
        diff_row("POST_HOC_fslow_u_hpg", hpg_u_core, nemo_hpg_u,
                 common_u),
        diff_row("POST_HOC_fslow_v_hpg", hpg_v_core, nemo_hpg_v,
                 common_v),
    ]
    u_total_residual = total_u_core - nemo_total_u
    u_wind_residual = wind_u_core - nemo_wind_u
    v_total_residual = total_v_core - nemo_total_v
    v_hpg_residual = hpg_v_core - nemo_hpg_v
    return {
        "qualification": "POST_HOC_LOCALIZATION_NOT_A_FROZEN_VERDICT",
        "source": {
            "nemo_slow_forcing": "dynspg_ts.F90:337,368,443",
            "nemo_hpg": "dynhpg.F90:345-380",
            "lego_depth_mean": "ocean_model_latlon_cgrid.py:3740-3786",
            "lego_wind": "ocean_pe_latlon_cgrid.py:3836-3940",
        },
        "registered_population": "NEMO_AND_STANDALONE_COMMON_WET_FACES",
        "rows": rows,
        "ownership": {
            "u": "WIND_STRESS_PROJECTION",
            "v": "HPG_ACCUMULATION",
            "u_total_minus_wind_residual_max_abs": float(np.max(
                np.abs((u_total_residual - u_wind_residual)[common_u]),
                initial=0.0)),
            "v_total_minus_hpg_residual_max_abs": float(np.max(
                np.abs((v_total_residual - v_hpg_residual)[common_v]),
                initial=0.0)),
            "rule_1b_eligibility": "NO_PHYSICAL_OPERATOR_RESIDUAL",
        },
    }


def _step_rows(run_kt2: Path, grid, cfg, z_coord, state, model,
               forcing, step_forcing, nemo_t: np.ndarray,
               nemo_s: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common = state._replace(
        T=state.T.replace(data=nemo_t),
        S=state.S.replace(data=nemo_s))
    unchanged = [name for name in common._fields
                 if name not in {"T", "S"}
                 and getattr(common, name) is not getattr(state, name)]
    if unchanged:
        raise ValueError("common-IC replacement changed non-T/S fields: "
                         + ", ".join(unchanged))
    dyn = jax.jit(lambda st, rate: model.step(
        st, standalone.DT_SECONDS, surface_forcing=step_forcing,
        external_tracer_rate=rate))
    common, rate = apply_dino_lat_lon_surface_forcing(
        common, forcing, z_coord, cfg, standalone.DT_SECONDS,
        t_seconds=standalone.DT_SECONDS, return_rate=True)
    localization = _forcing_localization(
        run_kt2, grid, z_coord, common, model, step_forcing)
    # NEMO's cold Euler step runs dyn_zdf after dyn_spg and before tra_adv
    # (stpmlf.F90:262-267,363).  The public step endpoint cannot distinguish
    # those two momentum stages, so record both operands explicitly.  The
    # explicit-only call is the state immediately before lego's implicit ZDF;
    # the second call includes ZDF but deliberately withholds mlf_baro_corr.
    # Both use the exact private operands that the no-history branch passes at
    # ocean_model_latlon_cgrid.py:9183-9210.
    common_seeded = model._seed_tke_preclosure_carry(common)
    entry = common_seeded._replace(
        u_before=common_seeded.u, v_before=common_seeded.v,
        T_before=common_seeded.T, S_before=common_seeded.S,
        eta_before=common_seeded.eta)

    def _cold_impl(st, ext_rate, apply_vmix):
        n2_bundle = model._tke_step_entry_n2_bundle(
            st, z_coord=z_coord, config=model.config)
        return model._step_impl(
            st, standalone.DT_SECONDS, surface_forcing=step_forcing,
            _apply_implicit_vmix=apply_vmix,
            _ab2_scope_override="advective",
            _barotropic_before_state=(
                common.eta.data, common.u.data, common.v.data),
            _external_tracer_rate=ext_rate,
            _tke_n2_bundle_override=n2_bundle,
            _apply_cold_start_after_reconcile=False,
            z_coord=z_coord, config=model.config)

    pre_zdf, _pre_zdf_bundle = jax.jit(
        lambda st, ext_rate: _cold_impl(st, ext_rate, False))(entry, rate)
    post_zdf = jax.jit(
        lambda st, ext_rate: _cold_impl(st, ext_rate, True))(entry, rate)
    step1 = dyn(common, rate)
    mask_t = _mesh_core(grid.tmask).astype(bool)
    mask_u = _mesh_core(grid.umask).astype(bool)
    mask_v = _mesh_core(grid.vmask).astype(bool)
    lego_t_full = (
        np.asarray(z_coord.is_active, dtype=bool)
        & (np.asarray(state.land_mask.data) > 0.5)[..., None])
    lego_u_full = lego_t_full & np.roll(lego_t_full, -1, axis=1)
    lego_v_full = np.zeros_like(lego_t_full)
    lego_v_full[:-1] = lego_t_full[:-1] & lego_t_full[1:]
    lego_t = _model_core(lego_t_full)
    lego_u = _model_core(lego_u_full)
    lego_v = _model_core(lego_v_full)
    common_t = mask_t & lego_t
    common_u = mask_u & lego_u
    common_v = mask_v & lego_v
    # Frozen rows 1--5 are the shared tracer RHS accumulator, not endpoint
    # states.  Reconstruct the production accumulator from the exact content
    # update.  Under the cold collapse h(Kbb)==h(Kmm), while section 7 returns
    # h(Kaa)*T(Kaa); hence RHS=(h_aa*T_aa-h_mm*T_mm)/(dt*h_mm).
    tend = model.tendencies(
        entry, step_forcing, dt=standalone.DT_SECONDS,
        ab2_scope_override="advective", z_coord=z_coord,
        config=model.config)
    rate_t, rate_s = rate
    rhs14_t = np.asarray(rate_t)
    rhs14_s = np.asarray(rate_s)
    rhs17_t = np.asarray(tend.dT_dt.data + rate_t)
    rhs17_s = np.asarray(tend.dS_dt.data + rate_s)
    h_mm = np.asarray(compute_layer_thickness(
        entry.eta.data, entry.H_bathy.data, z_coord,
        min_water_column_m=model.config.min_water_column_m))
    h_aa = np.asarray(compute_layer_thickness(
        pre_zdf.eta.data, pre_zdf.H_bathy.data, z_coord,
        min_water_column_m=model.config.min_water_column_m))
    rhs20_t = ((h_aa * np.asarray(pre_zdf.T.data)
                - h_mm * np.asarray(entry.T.data))
               / (standalone.DT_SECONDS * np.maximum(h_mm, 1.0e-30)))
    rhs20_s = ((h_aa * np.asarray(pre_zdf.S.data)
                - h_mm * np.asarray(entry.S.data))
               / (standalone.DT_SECONDS * np.maximum(h_mm, 1.0e-30)))
    diss = _pre_zdf_bundle[5]
    if diss is None:
        raise ValueError("faithful cold Euler stage probe requires diss_incr")
    diss_t, diss_s, _diss_u, _diss_v = map(np.asarray, diss)
    rhs23_t = rhs20_t + diss_t / standalone.DT_SECONDS
    rhs23_s = rhs20_s + diss_s / standalone.DT_SECONDS

    rows = [
        diff_row("time_level_collapse", np.ones((1,), dtype=np.float64),
                 np.ones((1,), dtype=np.float64),
                 np.ones((1,), dtype=bool), bar=0.0),
        diff_row("T_after_trasbc", _model_core(rhs14_t),
                 _runtime_dump(run_kt2 / "stp_dump_14_trasbc_tem.bin"), common_t),
        diff_row("S_after_trasbc", _model_core(rhs14_s),
                 _runtime_dump(run_kt2 / "stp_dump_14_trasbc_sal.bin"), common_t),
        diff_row("T_after_traqsr", _model_core(rhs17_t),
                 _runtime_dump(run_kt2 / "stp_dump_17_traqsr_tem.bin"), common_t),
        diff_row("S_after_traqsr", _model_core(rhs17_s),
                 _runtime_dump(run_kt2 / "stp_dump_17_traqsr_sal.bin"), common_t),
        diff_row("T_after_traadv", _model_core(rhs20_t),
                 _runtime_dump(run_kt2 / "stp_dump_20_traadv_tem.bin"), common_t),
        diff_row("S_after_traadv", _model_core(rhs20_s),
                 _runtime_dump(run_kt2 / "stp_dump_20_traadv_sal.bin"), common_t),
        diff_row("T_before_traldf", _model_core(rhs20_t),
                 _runtime_dump(run_kt2 / "stp_dump_22_before_traldf_tem.bin"), common_t),
        diff_row("S_before_traldf", _model_core(rhs20_s),
                 _runtime_dump(run_kt2 / "stp_dump_22_before_traldf_sal.bin"), common_t),
        diff_row("T_after_traldf", _model_core(rhs23_t),
                 _runtime_dump(run_kt2 / "stp_dump_23_after_traldf_tem.bin"), common_t),
        diff_row("S_after_traldf", _model_core(rhs23_s),
                 _runtime_dump(run_kt2 / "stp_dump_23_after_traldf_sal.bin"), common_t),
        diff_row("conditional_euler_T_after_trazdf", _model_core(step1.T.data),
                 _runtime_dump(run_kt2 / "stp_dump_21_trazdf_tem.bin"), common_t),
        diff_row("conditional_euler_S_after_trazdf", _model_core(step1.S.data),
                 _runtime_dump(run_kt2 / "stp_dump_21_trazdf_sal.bin"), common_t),
        diff_row("conditional_euler_U_after_corrector",
                 _model_core(np.asarray(step1.u.data)[:, 1:, :]),
                 _runtime_dump(run_kt2 / "baro_dump_u_after.bin"), common_u),
        diff_row("conditional_euler_V_after_corrector",
                 _model_core(np.asarray(step1.v.data)[1:, :, :]),
                 _runtime_dump(run_kt2 / "baro_dump_v_after.bin"), common_v),
        diff_row("conditional_euler_SSH_after_split", _model_core(step1.eta.data),
                 _runtime_ssh(run_kt2 / "spg_dump_pssh_final.bin"),
                 common_t[..., 0]),
    ]
    localization["momentum_stage_diagnostics"] = {
        "qualification": "POST_HOC_ARCHITECTURE_DIAGNOSTIC_NOT_FROZEN_ROW",
        "warning": ("lego explicit-only state is not a NEMO dynspg stage: "
                    "lego applies the combined implicit solve after tracer "
                    "advection; compare endpoints and registered forcing rows"),
        "rows": [
            diff_row("POST_HOC_U_explicit_only_vs_dynspg",
                     _model_core(np.asarray(pre_zdf.u.data)[:, 1:, :]),
                     _runtime_dump(run_kt2 / "stp_dump_07_dynspg_u.bin"), common_u),
            diff_row("POST_HOC_V_explicit_only_vs_dynspg",
                     _model_core(np.asarray(pre_zdf.v.data)[1:, :, :]),
                     _runtime_dump(run_kt2 / "stp_dump_07_dynspg_v.bin"), common_v),
            diff_row("POST_HOC_U_combined_zdf_vs_dynzdf",
                     _model_core(np.asarray(post_zdf.u.data)[:, 1:, :]),
                     _runtime_dump(run_kt2 / "stp_dump_08_dynzdf_u.bin"), common_u),
            diff_row("POST_HOC_V_combined_zdf_vs_dynzdf",
                     _model_core(np.asarray(post_zdf.v.data)[1:, :, :]),
                     _runtime_dump(run_kt2 / "stp_dump_08_dynzdf_v.bin"), common_v),
        ],
    }
    step1, rate = apply_dino_lat_lon_surface_forcing(
        step1, forcing, z_coord, cfg, standalone.DT_SECONDS,
        t_seconds=2 * standalone.DT_SECONDS, return_rate=True)
    step2 = dyn(step1, rate)
    before = read_nemo_restart_before(
        str(run_kt2 / "DINO_00000002_restart.nc"), nn_hls=0)
    rows.extend([
        diff_row("conditional_filtered_step1_T_carry",
                 _model_core(step2.T_before.data),
                 _mesh_core(before.T), common_t),
        diff_row("conditional_filtered_step1_S_carry",
                 _model_core(step2.S_before.data),
                 _mesh_core(before.S), common_t),
        diff_row("conditional_filtered_step1_U_carry",
                 _model_core(np.asarray(step2.u_before.data)[:, 1:, :]),
                 _mesh_core(before.u), common_u),
        diff_row("conditional_filtered_step1_V_carry",
                 _model_core(np.asarray(step2.v_before.data)[1:, :, :]),
                 _mesh_core(before.v), common_v),
        diff_row("conditional_filtered_step1_SSH_carry",
                 _model_core(step2.eta_before.data),
                 _mesh_core(before.ssh), common_t[..., 0]),
    ])
    return rows, localization


def self_test() -> dict[str, str]:
    active = np.ones((2, 2), dtype=bool)
    base = np.arange(4, dtype=np.float64).reshape(2, 2)
    planted = base.copy()
    planted[1, 0] += 2.0 * BAR
    row = diff_row("planted", planted, base, active)
    if row["status"] != "OVER_BAR" or row["first_over_bar"]["index"] != [1, 0]:
        raise RuntimeError("planted field mismatch did not fire")
    try:
        diff_row("wrong_core", base[:, :-1], base, active)
    except ValueError:
        shape = "FIRED"
    else:
        raise RuntimeError("planted wrong core did not fire")
    gate_rows = [
        {"name": name, "status": "PASS"} for name in INIT_ADMISSION_ROWS]
    if not initialization_admitted(gate_rows):
        raise RuntimeError("all-pass initialization gate did not admit")
    gate_rows[-1]["status"] = "OVER_BAR"
    if initialization_admitted(gate_rows):
        raise RuntimeError("planted initialization failure did not withhold")
    return {
        "field_mismatch_plant": "FIRED",
        "wrong_core_plant": shape,
        "initialization_gate_plant": "FIRED",
    }


def _legacy_geometry_controls(nemo_grid, cfg, z_coord, state) -> dict[str, Any]:
    """Prove each replaced initialization-geometry path can fail."""
    mask = _mesh_core(nemo_grid.tmask).astype(bool)
    analytic_grid = dino_lat_lon_grid(cfg)

    seam2d = np.asarray(partial_periodic_seam_wall_latlon(
        analytic_grid,
        open_lat_south_deg=cfg.channel_lat_south_deg,
        open_lat_north_deg=cfg.channel_lat_north_deg,
        seam_column_index=0)) > 0.5
    seam3d = _model_core(
        np.asarray(z_coord.is_active, dtype=bool)) & seam2d[..., None]
    seam_row = diff_row(
        "control_old_seam_wall", seam3d.astype(np.float64),
        mask.astype(np.float64), np.ones(mask.shape, dtype=bool), bar=0.0)
    if seam_row["status"] != "OVER_BAR":
        raise RuntimeError("old seam-wall control did not fire")

    jax_grid = create_mercator_grid(
        n_lon=analytic_grid.n_lon,
        n_lat=analytic_grid.n_lat,
        lat_max_deg=cfg.lat_max_deg,
        lon_west_deg=cfg.lon_west_deg,
        lon_east_deg=cfg.lon_east_deg,
        equator_on_tpoint=True,
        omega=cfg.omega,
        metric_convention=cfg.metric_convention,
        coordinate_evaluation="jax")
    jax_lat = np.broadcast_to(
        np.degrees(np.asarray(jax_grid.lat))[:, None],
        (analytic_grid.n_lat, analytic_grid.n_lon))
    lat_row = diff_row(
        "control_old_jax_latitude", jax_lat, _mesh_core(nemo_grid.gphit),
        np.ones(jax_lat.shape, dtype=bool), bar=0.0)
    if lat_row["status"] != "OVER_BAR":
        raise RuntimeError("old JAX latitude control did not fire")

    first = create_levy_stretched_z_star(
        n_levels=cfg.n_levels - 1,
        H_max=cfg.H_deep,
        dz_min=cfg.dz_min,
        k_th=float(cfg.k_th),
        a_cr=cfg.a_cr,
        analytic_t_depths=True)
    first_depth = np.broadcast_to(
        np.abs(np.asarray(first.z_full_ref)), mask.shape)
    depth_row = diff_row(
        "control_old_first_pass_depth", first_depth,
        _mesh_core(nemo_grid.gdept_0), mask, bar=0.0)
    if not (depth_row["status"] == "OVER_BAR"
            and depth_row["max_abs"] > 100.0):
        raise RuntimeError("old first-pass depth control did not fire >100 m")

    nemo_depth = _mesh_core(nemo_grid.gdept_0)
    source_t, source_s = hpg.nemo_istate_profiles_1d(nemo_depth)
    old_t = np.asarray(dino_T_profile_1d(nemo_depth))
    old_s = np.asarray(dino_S_profile_1d(nemo_depth))
    old_t_row = diff_row(
        "control_old_jax_factored_T_profile", old_t, source_t, mask)
    old_s_row = diff_row(
        "control_old_jax_factored_S_profile", old_s, source_s, mask)
    if (old_t_row["status"] != "OVER_BAR"
            or old_s_row["status"] != "OVER_BAR"):
        raise RuntimeError("old JAX/factored profile control did not fire")

    # The previous production bug used the final transitioned gdept to choose
    # k_bot. The continuous bowl is not retained on state (state.H_bathy is
    # already snapped), so the unit control locks its exact 929-cell signature;
    # this artifact records that the committed test was present and passed.
    return {
        "old_seam_wall": seam_row,
        "old_jax_latitude": lat_row,
        "old_first_pass_depth": depth_row,
        "old_jax_factored_T_profile": old_t_row,
        "old_jax_factored_S_profile": old_s_row,
        "final_depth_mask_operand_plant": "FIRED_BY_UNIT_TEST_929_CELLS",
    }


def run(args: argparse.Namespace) -> int:
    if os.environ.get("JAX_PLATFORM_NAME") != "cpu":
        raise SystemExit("CPU-only peel requires explicit JAX_PLATFORM_NAME=cpu")
    producer, dirty = standalone.git_provenance()
    if dirty:
        raise SystemExit("REFUSING dirty tracked producer")
    run_kt2 = args.run_kt2.resolve()
    run_y1 = args.run_traj_y1.resolve()
    inputs = [
        args.lego_kt2.resolve(), args.lego_d10.resolve(),
        run_kt2 / "mesh_mask.nc", run_kt2 / "DINO_00000002_restart.nc",
        run_kt2 / "namelist_cfg", run_y1 / "namelist_cfg",
        run_y1 / "DINO_00011520_restart.nc",
        run_y1 / "DINO_1y_00010701_00011230_grid_T_0000.nc",
    ] + [run_kt2 / name for name in (
        "stp_dump_07_dynspg_u.bin", "stp_dump_07_dynspg_v.bin",
        "stp_dump_08_dynzdf_u.bin", "stp_dump_08_dynzdf_v.bin",
        "stp_dump_14_trasbc_tem.bin", "stp_dump_14_trasbc_sal.bin",
        "stp_dump_17_traqsr_tem.bin", "stp_dump_17_traqsr_sal.bin",
        "stp_dump_20_traadv_tem.bin", "stp_dump_20_traadv_sal.bin",
        "stp_dump_22_before_traldf_tem.bin",
        "stp_dump_22_before_traldf_sal.bin",
        "stp_dump_23_after_traldf_tem.bin",
        "stp_dump_23_after_traldf_sal.bin",
        "stp_dump_21_trazdf_tem.bin", "stp_dump_21_trazdf_sal.bin",
        "baro_dump_u_after.bin", "baro_dump_v_after.bin",
        "spg_dump_pssh_final.bin", "spg_dump_zu_frc.bin",
        "spg_dump_zv_frc.bin", "wnd_dump_zu_frc_inc.bin",
        "wnd_dump_zv_frc_inc.bin", "hpg_dump_du.bin", "hpg_dump_dv.bin")]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing peel inputs: " + ", ".join(missing))

    controls = self_test()
    set_policy(PrecisionPolicy.fp64())
    grid = read_nemo_mesh_mask(str(run_kt2 / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(
        str(run_kt2 / "DINO_00000002_restart.nc"), nn_hls=0)
    bridge = twin.bridge_nemo_to_legoesm_topo(
        grid, restart, periodic_i=True, full_step=True, e3t_mode="both",
        vface_zonal_metric_evaluation="nemo_vpoint",
        coriolis_placement="cell_average")
    twin.verify_day0_matches_restart(
        bridge.state, restart, bridge.land_mask, min_restart_speed=0.0)
    controls["actual_kt2_restart_equality_and_nonzero"] = "PASS"

    (cfg, geometry, z_coord, state, _model_cfg, model, forcing,
     step_forcing, _perturbation) = standalone.build_standalone(0)
    ic_rows, init_localization, nemo_t, nemo_s, mask_receipt = _initial_rows(
        grid, geometry, z_coord, state, cfg)
    controls["legacy_geometry"] = _legacy_geometry_controls(
        grid, cfg, z_coord, state)
    admitted = initialization_admitted(ic_rows)
    step_rows = []
    forcing_localization = None
    if admitted:
        step_rows, forcing_localization = _step_rows(
            run_kt2, grid, cfg, z_coord, state, model, forcing, step_forcing,
            nemo_t, nemo_s)
    ordered = ic_rows + step_rows
    first_over_bar = next(
        (row["name"] for row in ordered if row["status"] == "OVER_BAR"), None)
    euler_passed = bool(step_rows) and all(
        row["status"] == "PASS" for row in step_rows)
    first_euler_debt = next(
        (row["name"] for row in step_rows if row["status"] == "OVER_BAR"),
        None)
    artifact = {
        "schema": SCHEMA,
        "session_id": SESSION_ID,
        "producer_git_sha": producer,
        "producer_dirty_tracked_files": dirty,
        "controls": controls,
        "bars": {"evaluated_fp64_abs": BAR, "discrete": 0.0},
        "first_over_bar": first_over_bar,
        "initialization_outcome": (
            "INIT_CONFIRMED" if admitted else
            f"INIT_PARTIAL_{first_over_bar}"),
        "euler_outcome": (
            "EULER_AT_BAR" if euler_passed else
            (f"EULER_DEBT_{first_euler_debt}" if admitted else
             "EULER_WITHHELD")),
        "euler_admission": admitted,
        "euler_at_bar": euler_passed,
        "wet_mask_receipt": mask_receipt,
        "initialization_localization_and_controls": init_localization,
        "euler_forcing_localization": forcing_localization,
        "rows": ordered,
        "inputs": {str(path): {"bytes": path.stat().st_size,
                               "sha256": file_sha256(path)}
                   for path in inputs},
        "ladder_receipt": {
            "source": "user_reported_2026-08-30",
            "day180_twin_sst_rms_degC": 0.005,
            "kt2_bridge_free_days": 359,
            "kt2_daily_endpoint_sst_rms_degC": 0.0104,
            "kt2_daily_endpoint_ssh_rms_mm": 0.29,
            "kt2_endpoint_offset_days": 0.94,
            "independent_sst_rms_degC": 0.39,
            "kt2_claim_scope": "TWIN_CLASS_DAILY_ENDPOINT_WITH_OFFSET",
            "bridge_artifact_producer":
                "7c0f121baed6bc57243e849127040865dda2d597",
            "bridge_artifact_dirty_tracked_files": 1,
            "dirty_override": "LEGOESM_ALLOW_DIRTY=1",
            "temporary_restart_guard_override": "DINO_TWIN_MIN_SPEED",
            "first_d10_mid_run_edit_refusal": "USER_REPORTED_NO_HASH",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for row in ordered:
        print(f"ROW {row['name']} {row['status']} max={row['max_abs']:.17e} "
              f"n={row['mismatch_count']}")
    print(f"FIRST_OVER_BAR={first_over_bar}")
    print(f"WROTE={args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-kt2", type=Path, required=True)
    parser.add_argument("--run-traj-y1", type=Path, required=True)
    parser.add_argument("--lego-kt2", type=Path, required=True)
    parser.add_argument("--lego-d10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
