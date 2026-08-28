#!/usr/bin/env python
"""Ordered day-180 ZDF sweep, stopping at the first numeric divergence.

The measurement contract and bars are frozen in ``PREREG_zdf_chain_sweep.md``.
This first-round probe deliberately implements only rows 1--3: it must stop
there if ``bn2`` diverges, rather than use downstream composites to skip over
the first failed operation.  Existing campaign loaders/state construction are
imported from ``eos_rab_bn2_per_element.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
if os.environ.get("DINO_1226_LANE") != "d180":
    raise SystemExit("Set DINO_1226_LANE=d180; this probe refuses every other lane")

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
ORACLE = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
RUN = ORACLE / "cfgs/DINO/RUN_SEQDUMP_D180_1R"
EXPECTED_MAP_SHA = "9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0"
EXPECTED_FOCUS = [(11, 1), (12, 1), (13, 1), (13, 23)]
POINTWISE_BAR = 1.0e-15
CORR_BAR = 1.0 - 1.0e-9
RATIO_EPS = 1.0e-6


def _import_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = _import_sibling("_zdf_bn2_base", "eos_rab_bn2_per_element.py")
kamm = _import_sibling("_zdf_kamm_twin", "kamm_twin_90d.py")
sh2_probe = _import_sibling("_zdf_sh2_capture", "sh2_canonical.py")
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    compute_buoyancy_frequency_nemo_bn2,
    nemo_bn2_depth_ladders,
    nemo_bn2_live_geometry,
    nemo_r3t_stretch,
    nemo_seos_alpha_beta,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_restart


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    override = os.environ.get("ZDF_SWEEP_TREE_SHA")
    if override is not None:
        if len(override) != 40 or any(c not in "0123456789abcdef" for c in override):
            raise SystemExit("ZDF_SWEEP_TREE_SHA must be a lowercase 40-hex commit")
        return override
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE.parents[3], text=True
    ).strip()


def focus_from_maps(path: Path) -> list[tuple[int, int]]:
    if sha256(path) != EXPECTED_MAP_SHA:
        raise SystemExit(f"MLD map SHA mismatch: {path}")
    with np.load(path) as maps:
        lego = maps["basin_legacy_base_index_day90"]
        nemo = maps["nemo_base_index_day90"]
        got = [tuple(map(int, x)) for x in np.argwhere(lego[:14] != nemo[:14])]
    if got != EXPECTED_FOCUS:
        raise SystemExit(f"focus registry changed: expected {EXPECTED_FOCUS}, got {got}")
    return got


def metrics(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray,
            focus: list[tuple[int, int]], bar: float) -> dict:
    registered = np.asarray(wet, dtype=bool)
    nonfinite = registered & (~np.isfinite(lego) | ~np.isfinite(nemo))
    finite_wet = registered & ~nonfinite
    ne = nemo[finite_wet]
    lo = lego[finite_wet]
    if ne.size == 0:
        raise AssertionError("empty wet comparison")
    scale = float(np.sqrt(np.mean(ne * ne)))
    if scale == 0.0:
        column = np.where(np.any(registered, axis=-1),
                          np.max(np.where(finite_wet, np.abs(lego - nemo), 0.0), axis=-1),
                          np.nan)
        threshold = 0.0
    else:
        column = np.where(np.any(registered, axis=-1),
                          np.max(np.where(finite_wet, np.abs(lego - nemo), 0.0), axis=-1) / scale,
                          np.nan)
        threshold = bar
    rms_ratio = float(np.sqrt(np.mean(lo * lo)) / scale) if scale else (
        1.0 if np.array_equal(lo, ne) else float("inf"))
    corr = float(np.corrcoef(lo, ne)[0, 1]) if np.std(lo) and np.std(ne) else None
    wet_columns = np.any(registered, axis=-1)
    bad = wet_columns & ((column > threshold) | np.any(nonfinite, axis=-1))
    focus_rows = []
    for j, i in focus:
        focus_rows.append({"j": j, "i": i, "wet": bool(wet_columns[j, i]),
                           "column_error": (float(column[j, i])
                                            if wet_columns[j, i] else None),
                           "pass": bool(wet_columns[j, i] and not bad[j, i])})
    aggregate_pass = ((corr is None or corr >= CORR_BAR)
                      and abs(rms_ratio - 1.0) <= RATIO_EPS
                      and not nonfinite.any())
    return {
        "n_wet_elements": int(registered.sum()),
        "n_nonfinite_wet_elements": int(nonfinite.sum()),
        "n_wet_columns": int(wet_columns.sum()),
        "n_diverged_columns": int(bad.sum()),
        "n_verified_columns": int(wet_columns.sum() - bad.sum()),
        "reference_rms": scale,
        "max_column_error": float(np.nanmax(column)),
        "worst_column_ji": [int(x) for x in np.unravel_index(np.nanargmax(column), column.shape)],
        "correlation": corr,
        "rms_ratio": rms_ratio,
        "focus": focus_rows,
        "bar": bar,
        "column_bar_pass": bool(not bad.any()),
        "aggregate_bar_pass": bool(aggregate_pass),
        "pass": bool(not bad.any() and aggregate_pass),
    }


def assemble_bn2(alpha, beta, T, S, gdept, gdepw, e3w):
    """NEMO eosbn2.F90:1459-1467 in the same expression order as production."""
    gd_up, gd_lo = gdept[..., :-1], gdept[..., 1:]
    zrw = (gdepw - gd_lo) / (gd_up - gd_lo)
    aw = alpha[..., 1:] * (1.0 - zrw) + alpha[..., :-1] * zrw
    bw = beta[..., 1:] * (1.0 - zrw) + beta[..., :-1] * zrw
    return NEMO_CONSTANTS_CONFIG.g * (
        aw * (T[..., :-1] - T[..., 1:])
        - bw * (S[..., :-1] - S[..., 1:])
    ) / e3w


def planted_controls(lego, nemo, wet, bar) -> dict:
    baseline = metrics(lego, nemo, wet, [], bar)
    if not baseline["pass"]:
        raise AssertionError("control requires a verified baseline row")
    idx = np.argwhere(wet)[0]
    planted = lego.copy()
    scale = baseline["reference_rms"]
    planted[tuple(idx)] += 100.0 * bar * scale
    poison = metrics(planted, nemo, wet, [], bar)
    rolled = np.roll(lego, 1, axis=1)
    roll = metrics(rolled, nemo, wet, [], bar)
    nonfinite = lego.copy()
    nonfinite[tuple(idx)] = np.nan
    nan_poison = metrics(nonfinite, nemo, wet, [], bar)
    if poison["pass"] or roll["pass"] or nan_poison["pass"]:
        raise AssertionError("a planted violation failed to fire")
    return {
        "baseline_pass": True,
        "perturbed_ji_k": [int(x) for x in idx],
        "perturbation_fired": not poison["pass"],
        "one_i_roll_fired": not roll["pass"],
        "nonfinite_fired": (not nan_poison["pass"]
                            and nan_poison["n_nonfinite_wet_elements"] == 1),
    }


def capture_face_sh2_call(model, state, forcing):
    """Capture production's exact zdfsh2 call operands and first result."""
    import legoesm.ocean.physics.vertical_mixing._shared as shared
    real = shared.avm_weighted_shear_production
    calls = []

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append((args, kwargs, out))
        return out

    shared.avm_weighted_shear_production = spy
    try:
        with jax.disable_jit():
            model.step(state, kamm.DT, surface_forcing=forcing)
    finally:
        shared.avm_weighted_shear_production = real
    if not calls:
        raise AssertionError("avm_weighted_shear_production never fired")
    return real, calls[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit(f"CPU/fp64 required; backend={jax.default_backend()} x64={jax.config.x64_enabled}")
    focus = focus_from_maps(args.mld_maps)
    st = base.build_state()
    jpi, jpj, jpk, hls = st["jpi"], st["jpj"], st["jpk"], st["hls"]
    nj, ni = st["ni_ni"]
    T, S = st["T"], st["S"]
    gdept, gdepw, e3w = nemo_bn2_live_geometry(
        st["z_coord"], st["eta"], st["H_bathy"])
    gdept = jnp.asarray(gdept, dtype=T.dtype)
    gdepw = jnp.asarray(gdepw, dtype=T.dtype)
    alpha, beta = nemo_seos_alpha_beta(T, S, gdept, NemoSEOSConfig())
    n2 = compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, NemoSEOSConfig(), g=NEMO_CONSTANTS_CONFIG.g,
        e3w_int=e3w)
    n2_legacy = compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, NemoSEOSConfig(), g=NEMO_CONSTANTS_CONFIG.g,
        e3w_source="depth_difference")

    alpha_n = base._load_haloed(str(RUN / "dump_alpha_b.bin"), jpi, jpj, hls)
    beta_n = base._load_haloed(str(RUN / "dump_beta_b.bin"), jpi, jpj, hls)
    n2_n_full = base._load_interior(str(RUN / "tke_dump_rn2b.bin"), ni, nj)
    gdept_n = base._load_haloed(str(RUN / "eiv_dump_gdept.bin"), jpi, jpj, hls)
    e3w_n_full = base._load_haloed(str(RUN / "eiv_dump_e3w.bin"), jpi, jpj, hls)

    active = np.asarray(st["active_3d"]) > 0.5
    wet_ab = active[..., :alpha_n.shape[-1]]
    wet_w_all = active[..., :-1] & active[..., 1:]
    row1a = metrics(np.asarray(alpha)[..., :alpha_n.shape[-1]], alpha_n,
                    wet_ab, focus, POINTWISE_BAR)
    row1b = metrics(np.asarray(beta)[..., :beta_n.shape[-1]], beta_n,
                    wet_ab, focus, POINTWISE_BAR)
    row1 = {"alpha": row1a, "beta": row1b,
            "disposition": "VERIFIED" if row1a["pass"] and row1b["pass"] else "DIVERGED"}
    controls = planted_controls(np.asarray(alpha)[..., :alpha_n.shape[-1]],
                                alpha_n, wet_ab, POINTWISE_BAR)

    n2_n = n2_n_full[..., 1:1 + n2.shape[-1]]
    row2m = metrics(np.asarray(n2), n2_n, wet_w_all, focus, POINTWISE_BAR)
    row2 = {"output": row2m,
            "legacy_depth_difference": metrics(
                np.asarray(n2_legacy), n2_n, wet_w_all, focus, POINTWISE_BAR),
            "disposition": "VERIFIED" if row2m["pass"] else "DIVERGED"}

    # Row 3: the same eos_rab/bn2 operation on Nnn tracers.  The restart's
    # tn/sn are the registered "now" operands; geometry remains Kmm=Nnn.
    now = read_nemo_restart(str(RUN / "DINO_00005760_restart.nc"), nn_hls=0)
    T_now = jnp.asarray(np.asarray(now.T).reshape(T.shape), dtype=T.dtype)
    S_now = jnp.asarray(np.asarray(now.S).reshape(S.shape), dtype=S.dtype)
    n2_now = compute_buoyancy_frequency_nemo_bn2(
        T_now, S_now, gdept, gdepw, NemoSEOSConfig(),
        g=NEMO_CONSTANTS_CONFIG.g, e3w_int=e3w)
    n2_now_n_full = base._load_interior(
        str(RUN / "tke_dump_rn2.bin"), ni, nj)
    n2_now_n = n2_now_n_full[..., 1:1 + n2_now.shape[-1]]
    row3m = metrics(np.asarray(n2_now), n2_now_n, wet_w_all, focus,
                    POINTWISE_BAR)
    row3 = {"output": row3m,
            "disposition": "VERIFIED" if row3m["pass"] else "DIVERGED"}

    row4 = {"disposition": "UNMEASURED",
            "reason": "an upstream row diverged"}
    sh2_localization = None
    if all(r["disposition"] == "VERIFIED" for r in (row1, row2, row3)):
        br, cfg, mc, model, forcing, sf, twin_state = kamm._build_twin_state(
            "nemo_dino_kamm_mlf", str(RUN), str(RUN), bridge_tke=True,
            bridge_before=True, restart_file="DINO_00005760_restart.nc",
            e3t_mode="both")
        tke_cfg = mc.physics.vertical_mixing.tke
        if (tke_cfg.tke_shear_production != "nemo_face_native"
                or tke_cfg.tke_shear_avm_weighting != "nemo_face"):
            raise AssertionError(
                "row-4 production path changed: expected face-native shear "
                "with face-averaged avm weighting")
        sh2_fn, (sh2_args, sh2_kwargs, sh2_prod) = capture_face_sh2_call(
            model, twin_state, sf)
        sh2_n_full = base._load_interior(
            str(RUN / "tke_dump_sh2.bin"), ni, nj)
        nsh = min(sh2_prod.shape[-1], sh2_n_full.shape[-1] - 1)
        sh2_n = sh2_n_full[..., 1:1 + nsh]
        wet_sh2 = wet_w_all[..., :nsh]
        row4m = metrics(np.asarray(sh2_prod)[..., :nsh], sh2_n, wet_sh2,
                        focus, POINTWISE_BAR)
        row4 = {"output": row4m,
                "disposition": "VERIFIED" if row4m["pass"] else "DIVERGED"}
        if row4["disposition"] == "DIVERGED":
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel,
            )
            cfg_legacy = tke_cfg._replace(tke_shear_avm_weighting="tpoint")
            mc_legacy = mc._replace(
                physics=mc.physics._replace(
                    vertical_mixing=mc.physics.vertical_mixing._replace(
                        tke=cfg_legacy)))
            model_legacy = LatLonCGridOceanModel(
                br.geometry, br.z_coord, mc_legacy)
            sh2_legacy = sh2_probe.run_and_capture_sh2(
                model_legacy, twin_state, sf, avm_weighting="tpoint")
            legacy_m = metrics(np.asarray(sh2_legacy)[..., :nsh], sh2_n,
                               wet_sh2, focus, POINTWISE_BAR)
            avm_n_full = base._load_interior(
                str(RUN / "tke_dump_avm_in.bin"), ni, nj)
            avm_n = avm_n_full[..., 1:1 + sh2_args[-1].shape[-1]]
            avm_operand_m = metrics(
                np.asarray(sh2_args[-1]), avm_n, wet_sh2, focus,
                POINTWISE_BAR)
            sh2_with_nemo_avm = sh2_fn(
                *sh2_args[:-1], jnp.asarray(avm_n), **sh2_kwargs)
            avm_sub_m = metrics(
                np.asarray(sh2_with_nemo_avm)[..., :nsh], sh2_n,
                wet_sh2, focus, POINTWISE_BAR)
            next_operand = None
            if avm_operand_m["pass"]:
                before = kamm.read_nemo_restart_before(
                    str(RUN / "DINO_00005760_restart.nc"), nn_hls=0)
                with xr.open_dataset(RUN / "mesh_mask.nc", decode_times=False) as ds:
                    umask = np.moveaxis(np.asarray(
                        ds["umask"].isel(time_counter=0)), 0, -1)
                    vmask = np.moveaxis(np.asarray(
                        ds["vmask"].isel(time_counter=0)), 0, -1)
                u_now_arg, v_now_arg, u_b_arg, v_b_arg = sh2_args[:4]
                u_now = np.asarray(u_now_arg)[:, 1:, :]
                v_now = np.asarray(v_now_arg)[1:, :, :]
                u_before = np.asarray(u_b_arg)[:, 1:, :]
                v_before = np.asarray(v_b_arg)[1:, :, :]
                velocity_identity = {
                    "u_now": metrics(u_now, now.u, umask > 0.5,
                                     focus, POINTWISE_BAR),
                    "v_now": metrics(v_now, now.v, vmask > 0.5,
                                     focus, POINTWISE_BAR),
                    "u_before_max_abs": float(np.max(np.abs(u_before - before.u))),
                    "v_before_max_abs": float(np.max(np.abs(v_before - before.v))),
                }
                now_sub = sh2_fn(
                    twin_state.u.data, twin_state.v.data,
                    *sh2_args[2:], **sh2_kwargs)
                now_sub_m = metrics(
                    np.asarray(now_sub)[..., :nsh], sh2_n, wet_sh2,
                    focus, POINTWISE_BAR)
                next_operand = {
                    "name": "NOW face-velocity differences at zdf_phy entry",
                    "nemo_line": "cfgs/DINO/WORK/zdfsh2.F90:81-82,86-87",
                    "legoesm_operand": "post-explicit/pre-implicit-solve u/v",
                    "velocity_operand_identity": velocity_identity,
                    "substitute_step_entry_now_velocities": now_sub_m,
                    "next_round_fix_design": {
                        "option": "tke_shear_evaluation_stage",
                        "faithful_default_on_complete_dino_nemo_cards":
                            "step_entry",
                        "legacy_opt_in": "implicit_solve_state",
                        "construction": "evaluate and carry p_sh2 at the "
                            "step-entry state, before explicit dynamics and "
                            "advection, then feed that frozen field to both "
                            "the TKE shear RHS and Prandtl denominator",
                    },
                }
            sh2_localization = {
                "nemo_operand_order": [
                    "face avm averages", "face velocity differences",
                    "e3uw/e3vw divisors", "four-face sum"],
                "fixed_operand": {
                    "name": "p_avm carried pre-step viscosity",
                    "nemo_line": "src/OCE/ZDF/zdfsh2.F90:80",
                    "metrics": avm_operand_m,
                },
                "first_diverging_operand": next_operand,
                "production_face_averaged_avm": row4m,
                "input_avm_current_subiteration_vs_nemo_carried": avm_operand_m,
                "substitute_nemo_carried_avm": avm_sub_m,
                "legacy_substitute_tpoint_avm": legacy_m,
                "first_passing_substitution": None,
            }

    localization = None
    if row1["disposition"] == "VERIFIED" and row2["disposition"] == "DIVERGED":
        # Exact dumps cover NEMO jk=1..35.  Score interfaces jk=2..35, i.e.
        # 34 legoESM interfaces; the omitted deepest interface has zero wet
        # cells on this DINO state and cannot affect the census.
        nk = min(alpha_n.shape[-1] - 1, e3w_n_full.shape[-1] - 1)
        sl = slice(0, nk + 1)
        wet = wet_w_all[..., :nk]
        gd = gdept[..., sl]
        gw = gdepw[..., :nk]
        TT, SS = T[..., sl], S[..., sl]
        aa, bb = alpha[..., sl], beta[..., sl]
        derived_e3w = gd[..., 1:] - gd[..., :-1]
        direct = assemble_bn2(aa, bb, TT, SS, gd, gw, derived_e3w)
        if not np.array_equal(np.asarray(direct), np.asarray(n2)[..., :nk]):
            raise AssertionError("localizer baseline is not bit-identical to production bn2")
        candidates = []
        def add(name, value):
            candidates.append({"substitution": name,
                               "metrics": metrics(np.asarray(value), n2_n[..., :nk],
                                                  wet, focus, POINTWISE_BAR)})
        # NEMO evaluation order: zrw geometry, alpha/beta interpolation,
        # T/S numerator, then the live-e3w divisor.
        restart_path = RUN / "DINO_00005760_restart.nc"
        with xr.open_dataset(restart_path, decode_times=False) as ds:
            tb_raw = np.moveaxis(np.asarray(ds["tb"].isel(time_counter=0)), 0, -1)
            sb_raw = np.moveaxis(np.asarray(ds["sb"].isel(time_counter=0)), 0, -1)
        t_identity = metrics(np.asarray(T), tb_raw, active, focus, POINTWISE_BAR)
        s_identity = metrics(np.asarray(S), sb_raw, active, focus, POINTWISE_BAR)
        gd_dump = jnp.asarray(gdept_n[..., :nk + 1], dtype=T.dtype)
        # mesh_mask carries gdepw_0/e3w_0 as independent NEMO operands.
        with xr.open_dataset(RUN / "mesh_mask.nc", decode_times=False) as ds:
            gdepw0_mesh = np.asarray(ds["gdepw_0"].isel(time_counter=0))
            gdept0_mesh = np.asarray(ds["gdept_0"].isel(time_counter=0))
            e3w0_mesh = np.asarray(ds["e3w_0"].isel(time_counter=0))
        gdepw0_mesh = jnp.asarray(np.moveaxis(gdepw0_mesh, 0, -1), dtype=T.dtype)
        gdept0_mesh = jnp.asarray(np.moveaxis(gdept0_mesh, 0, -1), dtype=T.dtype)
        e3w0_mesh = jnp.asarray(np.moveaxis(e3w0_mesh, 0, -1), dtype=T.dtype)
        stretch = nemo_r3t_stretch(st["z_coord"], st["eta"], st["H_bathy"])
        gdepw_mesh_live = gdepw0_mesh[..., 1:1 + nk] * stretch[..., None]
        gdepw_identity = metrics(np.asarray(gw), np.asarray(gdepw_mesh_live),
                                 wet, focus, POINTWISE_BAR)
        if not (t_identity["pass"] and s_identity["pass"]
                and gdepw_identity["pass"]):
            raise AssertionError(
                "operand identity precondition failed for T, S, or gdepw; "
                "refusing to attribute the first divergence to a later operand")
        add("mesh gdepw_0*(1+r3t) in zrw", assemble_bn2(
            aa, bb, TT, SS, gd, gdepw_mesh_live, derived_e3w))
        add("NEMO gdept(Kmm) in zrw", assemble_bn2(aa, bb, TT, SS, gd_dump, gw,
                                                   derived_e3w))
        add("NEMO dumped alpha/beta", assemble_bn2(
            jnp.asarray(alpha_n[..., :nk + 1]), jnp.asarray(beta_n[..., :nk + 1]),
            TT, SS, gd, gw, derived_e3w))
        # NEMO stores and stretches e3w as its own operand: e3w_0*(1+r3t).
        # This candidate is constructible in production without an oracle dump
        # and distinguishes that operation order from diff(gdept_0*stretch).
        gd_ref, _ = nemo_bn2_depth_ladders(st["z_coord"])
        e3w_ref = jnp.diff(jnp.asarray(gd_ref, dtype=T.dtype))[..., :nk]
        e3w_factorized = e3w_ref[None, None, :] * stretch[..., None]
        add("factorized e3w_0*(1+r3t) divisor", assemble_bn2(
            aa, bb, TT, SS, gd, gw, e3w_factorized))
        raw_mesh_diff = (gdept0_mesh[..., 1:nk + 1]
                         - gdept0_mesh[..., :nk])
        e3w_rawdiff_live = raw_mesh_diff * stretch[..., None]
        add("raw mesh diff(gdept_0)*(1+r3t) divisor", assemble_bn2(
            aa, bb, TT, SS, gd, gw, e3w_rawdiff_live))
        # Unlike NemoGrid today, mesh_mask.nc already carries NEMO's actual
        # 3-D e3w_0 operand.  Test the implementable design directly: preserve
        # that independent reference field and apply the live qco stretch.
        e3w_mesh_live = e3w0_mesh[..., 1:1 + nk] * stretch[..., None]
        add("mesh e3w_0*(1+r3t) divisor", assemble_bn2(
            aa, bb, TT, SS, gd, gw, e3w_mesh_live))
        e3w_dump = jnp.asarray(e3w_n_full[..., 1:1 + nk], dtype=T.dtype)
        add("NEMO live e3w(Kmm) divisor", assemble_bn2(aa, bb, TT, SS, gd, gw,
                                                       e3w_dump))
        first_passing = next((x["substitution"] for x in candidates
                              if x["metrics"]["pass"]), None)
        localization = {
            "scored_interfaces": nk,
            "omitted_deepest_wet_elements": int(wet_w_all[..., nk:].sum()),
            "input_identity": {"T_before_restart": t_identity,
                               "S_before_restart": s_identity,
                               "gdepw_mesh_live": gdepw_identity},
            "derived_e3w_vs_dump": metrics(np.asarray(derived_e3w),
                                            np.asarray(e3w_dump), wet, focus,
                                            POINTWISE_BAR),
            "factorized_e3w_vs_dump": metrics(np.asarray(e3w_factorized),
                                               np.asarray(e3w_dump), wet, focus,
                                               POINTWISE_BAR),
            "mesh_e3w0_live_vs_dump": metrics(np.asarray(e3w_mesh_live),
                                               np.asarray(e3w_dump), wet, focus,
                                               POINTWISE_BAR),
            "construction_diagnosis": {
                "raw_mesh_diff_gdept0_vs_e3w0": metrics(
                    np.asarray(raw_mesh_diff),
                    np.asarray(e3w0_mesh[..., 1:1 + nk]), wet, focus,
                    POINTWISE_BAR),
                "bridged_gdept_ref_vs_raw_mesh_gdept0": metrics(
                    np.asarray(jnp.broadcast_to(
                        jnp.asarray(gd_ref, dtype=T.dtype)[None, None, :nk + 1],
                        gdept0_mesh[..., :nk + 1].shape)),
                    np.asarray(gdept0_mesh[..., :nk + 1]),
                    active[..., :nk + 1], focus, POINTWISE_BAR),
                "raw_mesh_diff_live_vs_dump": metrics(
                    np.asarray(e3w_rawdiff_live), np.asarray(e3w_dump),
                    wet, focus, POINTWISE_BAR),
            },
            "candidates_in_nemo_evaluation_order": candidates,
            "first_passing_substitution": first_passing,
        }

    source_paths = [
        Path("packages/ocean/legoesm/ocean/eos.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/time_levels.py"),
        HERE / "dump_lane.py",
        HERE / "eos_rab_bn2_per_element.py",
        ORACLE / "cfgs/DINO/MY_SRC/stpmlf.F90",
        ORACLE / "cfgs/DINO/MY_SRC/zdftke.F90",
        ORACLE / "cfgs/DINO/WORK/zdfphy.F90",
        ORACLE / "cfgs/DINO/WORK/zdfsh2.F90",
        ORACLE / "src/OCE/TRA/eosbn2.F90",
        ORACLE / "src/OCE/ZDF/zdfsh2.F90",
    ]
    input_paths = [
        RUN / "DINO_00005760_restart.nc", RUN / "mesh_mask.nc", RUN / "ocean.output",
        RUN / "dump_alpha_b.bin", RUN / "dump_beta_b.bin", RUN / "tke_dump_rn2b.bin",
        RUN / "tke_dump_rn2.bin",
        RUN / "tke_dump_sh2.bin",
        RUN / "tke_dump_avm_in.bin",
        RUN / "eiv_dump_gdept.bin", RUN / "eiv_dump_e3w.bin", args.mld_maps,
    ]
    artifact = {
        "schema": "zdf-chain-sweep-v3",
        "lane": "d180", "kt": 5761, "cpu_only": True, "fp64": True,
        "checked_out_parent_sha": git_sha(),
        "probe_commit_sha": os.environ.get("ZDF_SWEEP_PROBE_SHA", "UNSTAMPED"),
        "platform": platform.platform(), "python": platform.python_version(),
        "jax": importlib.metadata.version("jax"), "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "time_levels": {x: time_level_for_dump(x) for x in
                        ("dump_alpha_b.bin", "dump_beta_b.bin", "tke_dump_rn2b.bin",
                         "tke_dump_rn2.bin",
                         "tke_dump_sh2.bin",
                         "tke_dump_avm_in.bin",
                         "eiv_dump_gdept.bin", "eiv_dump_e3w.bin")},
        "focus_columns_ji": [list(x) for x in focus],
        "bars": {"pointwise_column": POINTWISE_BAR, "corr": CORR_BAR,
                 "rms_ratio_epsilon": RATIO_EPS},
        "controls": controls,
        "rows": {"1_eos_rab_before": row1, "2_bn2_before": row2,
                 "3_eos_rab_bn2_now": row3,
                 "4_zdf_sh2": row4,
                 "5_bottom_drag_operands": {"disposition": "UNMEASURED",
                    "reason": "stop at row 4" if row4["disposition"] == "DIVERGED"
                    else "probe extension stops after row 4"}},
        "first_divergence": ({"row": 2, "operation": "bn2(Nbb)",
                              "nemo_line": "src/OCE/TRA/eosbn2.F90:1467",
                              "localization": localization}
                             if row2["disposition"] == "DIVERGED" else
                             ({"row": 3, "operation": "eos_rab/bn2(Nnn)",
                               "nemo_line": "src/OCE/TRA/eosbn2.F90:1467",
                               "localization": None}
                              if row3["disposition"] == "DIVERGED" else
                              ({"row": 4, "operation": "zdf_sh2",
                                "nemo_line": "src/OCE/ZDF/zdfsh2.F90:80-94",
                                "localization": sh2_localization}
                               if row4["disposition"] == "DIVERGED" else None))),
        "row4_localization": sh2_localization,
        "sha256": {str(p): sha256(p) for p in source_paths + input_paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"focus={focus}")
    print(f"row 1 eos_rab: {row1['disposition']} alpha_max={row1a['max_column_error']:.6e} "
          f"beta_max={row1b['max_column_error']:.6e}")
    print(f"row 2 bn2: {row2['disposition']} max={row2m['max_column_error']:.6e} "
          f"bad_columns={row2m['n_diverged_columns']}/{row2m['n_wet_columns']}")
    print(f"row 3 eos_rab/bn2 now: {row3['disposition']} "
          f"max={row3m['max_column_error']:.6e} "
          f"bad_columns={row3m['n_diverged_columns']}/{row3m['n_wet_columns']}")
    if row4["disposition"] != "UNMEASURED":
        row4m = row4["output"]
        print(f"row 4 zdf_sh2: {row4['disposition']} "
              f"max={row4m['max_column_error']:.6e} "
              f"bad_columns={row4m['n_diverged_columns']}/{row4m['n_wet_columns']}")
        if sh2_localization is not None:
            lm = sh2_localization["legacy_substitute_tpoint_avm"]
            print(f"  legacy substitute T-point avm: pass={lm['pass']} "
                  f"max={lm['max_column_error']:.6e} "
                  f"bad_columns={lm['n_diverged_columns']}/{lm['n_wet_columns']}")
            am = sh2_localization["substitute_nemo_carried_avm"]
            print(f"  substitute NEMO carried p_avm: pass={am['pass']} "
                  f"max={am['max_column_error']:.6e} "
                  f"bad_columns={am['n_diverged_columns']}/{am['n_wet_columns']}")
    if localization:
        for c in localization["candidates_in_nemo_evaluation_order"]:
            print(f"  substitute {c['substitution']}: pass={c['metrics']['pass']} "
                  f"max={c['metrics']['max_column_error']:.6e}")
        print(f"  first_passing_substitution={localization['first_passing_substitution']}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
