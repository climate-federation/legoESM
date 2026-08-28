#!/usr/bin/env python
"""Ordered day-180 ZDF sweep, stopping at the first numeric divergence.

The measurement contract and bars are frozen in ``PREREG_zdf_chain_sweep.md``
and its dated round amendments.  The probe extends only as each preceding row
crosses its registered bar and stops at the first failed operation.  Existing
campaign loaders/state construction are imported from
``eos_rab_bn2_per_element.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
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
if os.environ.get("DINO_1226_LANE") != "d180":
    raise SystemExit("Set DINO_1226_LANE=d180; this probe refuses every other lane")
_ENV_DEFAULTS = {
    "LEGOESM_NEMO_E3T": "both",
    "LEGOESM_FIDELITY_FP64": "1",
    "FP64": "1",
    "LEGOESM_VMIX_F32_SOLVE": "0",
    "LEGOESM_BAROCLINIC_F32": "0",
    "LEGOESM_TRACER_PAIR": "0",
    "LEGOESM_VMIX_BATCHED": "0",
    "LEGOESM_VMIX_TSPAIR": "1",
}
for _name, _expected in _ENV_DEFAULTS.items():
    _actual = os.environ.get(_name, _expected)
    if _actual != _expected:
        raise SystemExit(
            f"{_name}={_actual!r} is incompatible with this registered lane; "
            f"expected {_expected!r}")
    os.environ[_name] = _expected
for _name in (
    "DINO_EEN_METRIC", "DINO_BOLUS_ADV", "DINO_RECONCILE_TARGET",
    "DINO_AFTER_RECONCILE", "DINO_NEMO_KMM_DIVISOR",
    "DINO_OUTER_INTEGRATOR", "DINO_TWIN_SEASONAL_KT0",
):
    if _name in os.environ:
        raise SystemExit(
            f"{_name} is an ablation override and must be unset for this lane")

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
sh2_walk = _import_sibling("_zdf_sh2_walk", "sh2_walk.py")
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
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE.parents[3], text=True
    ).strip()


def probe_commit_sha() -> str:
    repo = HERE.parents[3]
    path = Path(__file__).resolve().relative_to(repo)
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=repo)
    if clean.returncode != 0:
        raise SystemExit(
            "zdf_chain_sweep.py differs from HEAD; refusing an unstamped run")
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=repo, text=True).strip()


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


def face_difference_metrics(
    lego_velocity: np.ndarray,
    nemo_velocity: np.ndarray,
    face_mask: np.ndarray,
    focus: list[tuple[int, int]],
    orientation: str,
) -> dict:
    """Score the literal zdf_sh2 vertical-difference operand on wet faces.

    The whole-domain population is one column per U or V face.  A registered
    focus item is a T column, so its focus score is the maximum over its two
    surrounding wet faces (west/east for U, south/north for V), matching the
    later four-face assembly rather than arbitrarily naming only one face.
    """
    lego_velocity = np.asarray(lego_velocity)
    nemo_velocity = np.asarray(nemo_velocity)
    face_mask = np.asarray(face_mask)
    if lego_velocity.shape != nemo_velocity.shape or face_mask.shape != nemo_velocity.shape:
        raise AssertionError(
            "velocity/mask shape mismatch for direct difference operand: "
            f"lego={lego_velocity.shape} nemo={nemo_velocity.shape} "
            f"mask={face_mask.shape}")
    lego_diff = lego_velocity[..., :-1] - lego_velocity[..., 1:]
    nemo_diff = nemo_velocity[..., :-1] - nemo_velocity[..., 1:]
    wet = (face_mask[..., :-1] > 0.5) & (face_mask[..., 1:] > 0.5)
    out = metrics(lego_diff, nemo_diff, wet, focus, POINTWISE_BAR)

    scale = out["reference_rms"]
    focus_rows = []
    ny, nx = wet.shape[:2]
    for j, i in focus:
        if orientation == "u":
            faces = ((j, (i - 1) % nx), (j, i % nx))
        elif orientation == "v":
            faces = ((max(j - 1, 0), i), (min(j, ny - 1), i))
        else:
            raise ValueError(f"unknown face orientation {orientation!r}")
        scored = []
        for fj, fi in faces:
            active = bool(np.any(wet[fj, fi]))
            err = (float(np.max(np.abs(
                lego_diff[fj, fi][wet[fj, fi]]
                - nemo_diff[fj, fi][wet[fj, fi]])) / scale)
                   if active and scale else (0.0 if active else None))
            scored.append({"j": int(fj), "i": int(fi), "wet": active,
                           "column_error": err,
                           "pass": bool(active and err <= POINTWISE_BAR)})
        active_scores = [x["column_error"] for x in scored if x["wet"]]
        focus_rows.append({
            "j": j, "i": i, "wet": bool(active_scores),
            "column_error": max(active_scores) if active_scores else None,
            "pass": bool(active_scores and max(active_scores) <= POINTWISE_BAR),
            "surrounding_faces": scored,
        })
    out["focus"] = focus_rows
    out["operand"] = "velocity(k-1)-velocity(k)"
    out["focus_semantics"] = "maximum over the two surrounding wet faces"
    return out


def assemble_sh2_with_metrics(
    u_now, v_now, u_before, v_before, avm,
    e3u_now, e3u_before, e3v_now, e3v_before,
    u_mask, v_mask,
):
    """Literal legoESM-face-index form of zdfsh2.F90:80-94.

    Separate NOW/BEFORE face metrics make lines 83/88 independently
    substitutable; keeping this in the committed probe prevents a metric
    diagnosis from being inferred from a downstream composite.
    """
    du_n = u_now[..., :-1] - u_now[..., 1:]
    du_b = u_before[..., :-1] - u_before[..., 1:]
    dv_n = v_now[..., :-1] - v_now[..., 1:]
    dv_b = v_before[..., :-1] - v_before[..., 1:]
    wumask = u_mask[..., :-1] * u_mask[..., 1:]
    wvmask = v_mask[..., :-1] * v_mask[..., 1:]
    km_u = (jnp.concatenate([avm[:, -1:, :], avm], axis=1)
            + jnp.concatenate([avm, avm[:, :1, :]], axis=1))
    km_v = (jnp.concatenate([avm[:1, :, :], avm], axis=0)
            + jnp.concatenate([avm, avm[-1:, :, :]], axis=0))
    z_u = km_u * du_n * du_b / (e3u_now * e3u_before) * wumask
    z_v = km_v * dv_n * dv_b / (e3v_now * e3v_before) * wvmask
    coast_u = 2.0 - u_mask[:, :-1, 1:] * u_mask[:, 1:, 1:]
    coast_v = 2.0 - v_mask[:-1, :, 1:] * v_mask[1:, :, 1:]
    return 0.25 * (
        (z_u[:, :-1, :] + z_u[:, 1:, :]) * coast_u
        + (z_v[:-1, :, :] + z_v[1:, :, :]) * coast_v)


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


def planted_point_controls(lego, nemo, wet, bar) -> dict:
    """Red controls for a zonally invariant field where an i-roll is inert."""
    baseline = metrics(lego, nemo, wet, [], bar)
    if not baseline["pass"]:
        raise AssertionError("point control requires a verified baseline row")
    idx = np.argwhere(wet)[0]
    planted = lego.copy()
    planted[tuple(idx)] += 100.0 * bar * baseline["reference_rms"]
    poison = metrics(planted, nemo, wet, [], bar)
    nonfinite = lego.copy()
    nonfinite[tuple(idx)] = np.nan
    nan_poison = metrics(nonfinite, nemo, wet, [], bar)
    if poison["pass"] or nan_poison["pass"]:
        raise AssertionError("a planted point violation failed to fire")
    return {
        "baseline_pass": True,
        "perturbed_ji_k": [int(x) for x in idx],
        "perturbation_fired": not poison["pass"],
        "nonfinite_fired": (not nan_poison["pass"]
                            and nan_poison["n_nonfinite_wet_elements"] == 1),
        "horizontal_roll": "WAIVED: DINO analytic wind is zonally invariant",
    }


def capture_face_sh2_call(model, state, forcing):
    """Capture production's exact zdfsh2 call operands and first result."""
    import legoesm.ocean.physics.vertical_mixing._shared as shared
    import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
    real = shared.avm_weighted_shear_production
    real_mxl = tke_mod.compute_mixing_lengths
    real_solve = tke_mod._solve_tke_backward_euler
    real_lc = tke_mod.nemo_langmuir_tke_source
    calls = []
    mxl_calls = []
    solve_calls = []
    lc_calls = []

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append((args, kwargs, out))
        return out

    def spy_mxl(*args, **kwargs):
        out = real_mxl(*args, **kwargs)
        mxl_calls.append((args, kwargs, out))
        return out

    def spy_solve(**kwargs):
        out = real_solve(**kwargs)
        solve_calls.append((kwargs, out))
        return out

    def spy_lc(*args, **kwargs):
        out = real_lc(*args, **kwargs)
        lc_calls.append((args, kwargs, out))
        return out

    shared.avm_weighted_shear_production = spy
    tke_mod.compute_mixing_lengths = spy_mxl
    tke_mod._solve_tke_backward_euler = spy_solve
    tke_mod.nemo_langmuir_tke_source = spy_lc
    try:
        with jax.disable_jit():
            model.step(state, kamm.DT, surface_forcing=forcing)
    finally:
        shared.avm_weighted_shear_production = real
        tke_mod.compute_mixing_lengths = real_mxl
        tke_mod._solve_tke_backward_euler = real_solve
        tke_mod.nemo_langmuir_tke_source = real_lc
    if not calls:
        raise AssertionError("avm_weighted_shear_production never fired")
    if not mxl_calls or not solve_calls or not lc_calls:
        raise AssertionError("TKE mixing-length/Langmuir/solve stages never fired")
    return real, calls[0], {
        "mxl_calls": mxl_calls, "lc_calls": lc_calls,
        "solve_calls": solve_calls,
    }


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
        sh2_fn, (sh2_args, sh2_kwargs, sh2_prod), tke_capture = capture_face_sh2_call(
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
        if row4["disposition"] == "VERIFIED":
            row4["controls"] = planted_controls(
                np.asarray(sh2_prod)[..., :nsh], sh2_n, wet_sh2,
                POINTWISE_BAR)
            from legoesm import constants
            from legoesm.ocean.dynamics.ocean_tendency_common import (
                nemo_effective_bottom_drag_r,
            )
            from legoesm.ocean.eos import make_eos_fn
            from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
                _nemo_mld_from_n2_integral, _nemo_native_active_3d,
                gm_redi_density_and_jacobian,
            )
            import legoesm.ocean.physics.vertical_mixing.tke as tke_mod

            wet2 = np.asarray(twin_state.land_mask.data) > 0.5
            bl = np.maximum(np.asarray(br.z_coord.bottom_level), 0)
            idx = bl[..., None]
            ucc = 0.5 * (twin_state.u.data[:, :-1, :]
                         + twin_state.u.data[:, 1:, :])
            vcc = 0.5 * (twin_state.v.data[:-1, :, :]
                         + twin_state.v.data[1:, :, :])
            u_bot = jnp.take_along_axis(ucc, idx, axis=-1)[..., 0]
            v_bot = jnp.take_along_axis(vcc, idx, axis=-1)[..., 0]
            h_bot = jnp.take_along_axis(
                br.z_coord.h_partial, idx, axis=-1)[..., 0]
            drag_cfg = mc.bottom_drag
            r_drag = nemo_effective_bottom_drag_r(
                u_bot, v_bot, h_bot,
                scheme=drag_cfg.bottom_drag_scheme,
                cd0=float(drag_cfg.bottom_drag_cd0),
                cd_max=float(drag_cfg.bottom_drag_cdmax),
                z0=float(drag_cfg.bottom_drag_z0),
                ke0=float(drag_cfg.bottom_drag_ke0),
                von_karman=constants.kappa_von_karman)
            rcdu = np.fromfile(
                RUN / "drg_dump_rCdU_bot.bin", dtype="<f8").reshape(jpj, jpi)
            rcdu = rcdu[hls:-hls, hls:-hls]
            row5m = metrics(np.asarray(r_drag)[..., None],
                            (-rcdu)[..., None], wet2[..., None],
                            focus, POINTWISE_BAR)

            before = kamm.read_nemo_restart_before(
                str(RUN / "DINO_00005760_restart.nc"), nn_hls=0)
            Tb = jnp.asarray(before.T, dtype=twin_state.T.data.dtype)
            Sb = jnp.asarray(before.S, dtype=twin_state.S.data.dtype)
            eos_fn = make_eos_fn(mc.eos, mc.eos_linear)
            _, jac_mld = gm_redi_density_and_jacobian(
                Tb, Sb, twin_state.eta.data, twin_state.H_bathy.data,
                br.geometry, br.z_coord, eos=mc.eos,
                eos_linear=mc.eos_linear, mask=twin_state.land_mask.data,
                rho_0=mc.rho_0, g=mc.g)
            active_mld = _nemo_native_active_3d(
                twin_state.land_mask.data, br.z_coord,
                twin_state.H_bathy.data, Tb.dtype)
            hml, mbase = _nemo_mld_from_n2_integral(
                Tb, Sb, twin_state.land_mask.data, br.z_coord, eos_fn,
                mc.gm_redi.mld_rho_c, mc.g, mc.rho_0,
                active_3d=active_mld, jacobian=jac_mld)
            nmln_n = base._load_haloed(
                str(RUN / "dump_nmln.bin"), jpi, jpj, hls)[..., 0]
            hmlp_n = base._load_haloed(
                str(RUN / "dump_hmlp.bin"), jpi, jpj, hls)[..., 0]
            row6_lego = np.asarray(mbase, dtype=np.int64) + 2
            row6_nemo = np.asarray(nmln_n, dtype=np.int64)
            integer_mismatch = wet2 & (row6_lego != row6_nemo)
            row6m = metrics(row6_lego[..., None], row6_nemo[..., None],
                            wet2[..., None], focus, 1.0e-12)
            row6m["integer_exact"] = bool(not integer_mismatch.any())
            row6m["n_integer_mismatch_columns"] = int(integer_mismatch.sum())
            row6m["registered_row_class"] = "A/E"
            _row6_poison = row6_lego.copy()
            _row6_idx = tuple(np.argwhere(wet2)[0])
            _row6_poison[_row6_idx] += 1
            row6m["off_by_one_control_fired"] = bool(
                np.any(wet2 & (_row6_poison != row6_nemo)))
            if not row6m["off_by_one_control_fired"]:
                raise AssertionError("row-6 integer off-by-one control did not fire")
            row6m["pass"] = bool(row6m["pass"] and not integer_mismatch.any())
            row7m = metrics(np.asarray(hml)[..., None], hmlp_n[..., None],
                            wet2[..., None], focus, POINTWISE_BAR)

            tke_cfg = mc.physics.vertical_mixing.tke
            surf = tke_mod._surface_tke_dirichlet(
                tke_cfg, sf.taum, mc.rho_0)
            zbbrau = np.float64(tke_mod._NEMO_TKE_EBB) / np.float64(mc.rho_0)
            utau_n = np.fromfile(
                RUN / "sbc_dump_utau.bin", dtype="<f8").reshape(jpj, jpi)
            utau_n = utau_n[hls:jpj - hls, hls:jpi - hls]
            with xr.open_dataset(RUN / "mesh_mask.nc", decode_times=False) as ds:
                gphiu = np.asarray(
                    ds["gphiu"].isel(time_counter=0), dtype=np.float64)
            lego_lat = np.broadcast_to(
                np.asarray(forcing["lat_deg_1d"], dtype=np.float64)[:, None],
                gphiu.shape)
            row8_gphiu = metrics(
                lego_lat[..., None], gphiu[..., None], wet2[..., None],
                focus, POINTWISE_BAR)

            nodes = np.asarray(cfg.wind_tau_lats_deg, dtype=np.float64)
            values = np.asarray(cfg.wind_tau_values, dtype=np.float64)
            utau_literal = np.empty_like(gphiu)
            for flat_index, phi in enumerate(gphiu.flat):
                zdphi = nodes - phi
                kmin = int(np.argmin(np.abs(zdphi)))
                if zdphi[kmin] <= 0.0:
                    ks, kn = kmin, kmin + 1
                else:
                    ks, kn = kmin - 1, kmin
                zs = min(1.0, max(
                    0.0, (phi - nodes[ks]) / (nodes[kn] - nodes[ks])))
                utau_literal.flat[flat_index] = (
                    values[ks]
                    + (values[kn] - values[ks]) * (3.0 - 2.0 * zs)
                    * zs ** 2)
            row8_utau_prod = metrics(
                np.asarray(forcing["tau_u_cell_2d"])[..., None],
                utau_n[..., None], wet2[..., None], focus, POINTWISE_BAR)
            row8_utau_literal = metrics(
                utau_literal[..., None], utau_n[..., None], wet2[..., None],
                focus, POINTWISE_BAR)
            from legoesm.ocean.experiments.dino import (
                dino_lat_lon_surface_forcing_arrays,
            )
            legacy_cfg = dataclasses.replace(
                cfg, dino_wind_profile_evaluation="factored_smoothstep")
            legacy_utau = np.asarray(dino_lat_lon_surface_forcing_arrays(
                br.geometry, legacy_cfg,
                wind_lat_deg=gphiu[:, 0])["tau_u_cell_2d"])
            row8_legacy_utau = metrics(
                legacy_utau[..., None], utau_n[..., None], wet2[..., None],
                focus, POINTWISE_BAR)
            taum_dump = np.abs(utau_n)
            taum_dump = np.where(utau_n > 0.0, taum_dump * 1.3, taum_dump)
            surface_n = np.maximum(
                np.float64(tke_mod._NEMO_TKE_EMIN0), zbbrau * taum_dump)
            row8m = metrics(np.asarray(surf)[..., None],
                            surface_n[..., None], wet2[..., None], focus,
                            POINTWISE_BAR)
            row8_taum_dump = metrics(
                np.asarray(sf.taum)[..., None], taum_dump[..., None],
                wet2[..., None], focus, POINTWISE_BAR)
            observable = wet2 & (surface_n > tke_mod._NEMO_TKE_EMIN0)
            row8_taum = metrics(
                np.asarray(sf.taum)[..., None], taum_dump[..., None],
                observable[..., None], focus, POINTWISE_BAR)
            literal = np.maximum(
                np.float64(tke_mod._NEMO_TKE_EMIN0),
                zbbrau * np.asarray(sf.taum))
            row8_literal = metrics(
                literal[..., None], surface_n[..., None], wet2[..., None],
                focus, POINTWISE_BAR)
            row8_surface_sub = metrics(
                surface_n[..., None], surface_n[..., None],
                wet2[..., None], focus, POINTWISE_BAR)
            row8_controls = planted_point_controls(
                surface_n[..., None], surface_n[..., None],
                wet2[..., None], POINTWISE_BAR)
            row8m["controls"] = row8_controls
            en_full = base._load_interior(
                str(RUN / "tke_dump_en.bin"), ni, nj)
            row8_poststage_invariant = metrics(
                surface_n[..., None], en_full[..., :1], wet2[..., None],
                focus, POINTWISE_BAR)
            if not row8_poststage_invariant["pass"]:
                raise AssertionError(
                    "post-tke_tke en(1) changed after the row-8 assignment")
            if row8m["pass"]:
                if row8_legacy_utau["pass"]:
                    raise AssertionError(
                        "row-8 legacy profile control did not reproduce the "
                        "registered sbc_dump_utau.bin failure")
                if not row8_utau_literal["pass"]:
                    raise AssertionError(
                        "row-8 production passed but the independent literal "
                        "wind reconstruction did not")
            row4["continuation_preview"] = {
                "rows": {
                    "5_bottom_drag_coefficient": row5m,
                    "6_native_mld_index": row6m,
                    "7_native_mld_depth": row7m,
                    "8_surface_tke_boundary": row8m,
                },
                "row8_legacy_dump_control": row8_legacy_utau,
            }
            if not row8m["pass"]:
                row4["continuation_preview"]["first_divergence"] = {
                    "row": 8,
                    "operation": "TKE surface Dirichlet boundary",
                    "nemo_line": "cfgs/DINO/MY_SRC/zdftke.F90:334,360-364",
                    "output": row8m,
                    "first_failing_operand": {
                        "name": "utau cubic-profile arithmetic association",
                        "nemo_line":
                            "cfgs/DINO/MY_SRC/usrdef_sbc.F90:221-223,632",
                        "legoesm_line":
                            "packages/ocean/legoesm/ocean/experiments/dino.py:2086-2088",
                    },
                    "operand_localization": {
                        "gphiu": row8_gphiu,
                        "production_utau_vs_dump": row8_utau_prod,
                        "nemo_literal_utau_vs_dump": row8_utau_literal,
                        "production_taum_vs_dump_derived": row8_taum_dump,
                        "production_taum_on_unfloored_columns": row8_taum,
                        "literal_zbbrau_times_taum_then_max": row8_literal,
                        "substitute_dump_derived_taum": row8_surface_sub,
                        "n_observable_unfloored_columns": int(observable.sum()),
                        "legacy_factored_utau_vs_dump": row8_legacy_utau,
                    },
                    "poststage_surface_en_invariant": row8_poststage_invariant,
                    "controls": row8_controls,
                    "next_round_fix_design": {
                        "option": "dino_wind_profile_evaluation",
                        "faithful_default_on_complete_dino_nemo_cards":
                            "nemo_literal",
                        "legacy_default_everywhere_else": "factored_smoothstep",
                        "legacy_opt_in_on_dino_nemo_cards":
                            "factored_smoothstep",
                        "construction": "preserve NEMO's nearest-node interval "
                            "selection and left-associated expression "
                            "tau_s + (tau_n-tau_s)*(3-2*s)*s**2 before ABS "
                            "and the conditional 1.3 westerly multiplier",
                    },
                }
            else:
                # Row 9, and no later row, is evaluated until this boundary
                # crosses its registered pointwise bar.  Reconstruct
                # zdftke.F90:377-384 from NEMO's own Kbb face velocities,
                # bottom masks/index and dumped rCdU_bot.
                solve_kwargs = tke_capture["solve_calls"][0][0]
                bottom_prod = np.asarray(solve_kwargs["bottom_dirichlet"])
                bottom_level = np.asarray(
                    solve_kwargs["bottom_level"], dtype=np.int64)
                with xr.open_dataset(
                        RUN / "mesh_mask.nc", decode_times=False) as ds:
                    umask_b = np.moveaxis(np.asarray(
                        ds["umask"].isel(time_counter=0)), 0, -1)
                    vmask_b = np.moveaxis(np.asarray(
                        ds["vmask"].isel(time_counter=0)), 0, -1)
                    # NEMO's mbkt is independently initialized from mbathy.
                    # mesh_mask stores 1-based deepest wet T levels, whereas
                    # legoESM bottom_level is 0-based.
                    nemo_bottom_level = np.asarray(
                        ds["mbathy"].isel(time_counter=0), dtype=np.int64)
                # mesh_mask.nc is already the model-domain 199x52 array;
                # unlike raw bracket streams it has no additional dump halo.
                nemo_bottom_level = nemo_bottom_level - 1
                bottom_index_mismatch = wet2 & (
                    bottom_level != nemo_bottom_level)
                bottom_index_identity = {
                    "pass": not bool(np.any(bottom_index_mismatch)),
                    "n_mismatch_wet_columns": int(
                        np.count_nonzero(bottom_index_mismatch)),
                    "n_wet_columns": int(np.count_nonzero(wet2)),
                    "nemo_source": "mesh_mask.nc:mbathy - 1",
                }
                if not bottom_index_identity["pass"]:
                    raise AssertionError(
                        "legoESM bottom_level != independent NEMO mbathy-1")
                kb = np.clip(nemo_bottom_level, 0,
                             before.u.shape[-1] - 1)[..., None]
                ub_i = np.take_along_axis(before.u, kb, axis=-1)[..., 0]
                ub_w = np.take_along_axis(
                    np.roll(before.u, 1, axis=1), kb, axis=-1)[..., 0]
                vb_i = np.take_along_axis(before.v, kb, axis=-1)[..., 0]
                vb_s = np.take_along_axis(
                    np.roll(before.v, 1, axis=0), kb, axis=-1)[..., 0]
                um_i = np.take_along_axis(umask_b, kb, axis=-1)[..., 0]
                um_w = np.take_along_axis(
                    np.roll(umask_b, 1, axis=1), kb, axis=-1)[..., 0]
                vm_i = np.take_along_axis(vmask_b, kb, axis=-1)[..., 0]
                vm_s = np.take_along_axis(
                    np.roll(vmask_b, 1, axis=0), kb, axis=-1)[..., 0]
                zmsku = 2.0 - um_w * um_i
                zmskv = 2.0 - vm_s * vm_i
                bottom_literal = np.maximum(
                    -np.float64(0.001875) * rcdu
                    * np.sqrt((zmsku * (ub_i + ub_w)) ** 2
                              + (zmskv * (vb_i + vb_s)) ** 2),
                    np.float64(tke_cfg.tke_background)) * wet2
                row9m = metrics(
                    bottom_prod[..., None], bottom_literal[..., None],
                    wet2[..., None], focus, POINTWISE_BAR)
                row9_controls = planted_controls(
                    bottom_literal[..., None], bottom_literal[..., None],
                    wet2[..., None], POINTWISE_BAR)
                index_poison = nemo_bottom_level.copy()
                poison_ji = tuple(np.argwhere(wet2)[0])
                index_poison[poison_ji] += 1
                bottom_index_identity["off_by_one_control_fired"] = bool(
                    index_poison[poison_ji] != bottom_level[poison_ji])
                if not bottom_index_identity["off_by_one_control_fired"]:
                    raise AssertionError(
                        "row-9 independent-index red control did not fire")
                row9m["bottom_index_identity"] = bottom_index_identity
                row9m["controls"] = row9_controls
                row4["continuation_preview"]["rows"][
                    "9_bottom_tke_boundary"] = row9m
                if not row9m["pass"]:
                    row4["continuation_preview"]["first_divergence"] = {
                        "row": 9,
                        "operation": "bottom TKE Dirichlet boundary",
                        "nemo_line":
                            "cfgs/DINO/MY_SRC/zdftke.F90:377-384",
                        "output": row9m,
                        "first_failing_operand": {
                            "name": "bottom boundary composite",
                            "nemo_line":
                                "cfgs/DINO/MY_SRC/zdftke.F90:378-383",
                        },
                        "operand_localization": {
                            "production_vs_literal": row9m,
                        },
                        "controls": row9_controls,
                    }
                else:
                    # Row 10: stop on the first dumped operand in the literal
                    # ln_lc chain, zdftke.F90:401-468.  The full source is not
                    # dispositioned without a NEMO stage dump; its offline
                    # reconstruction below is diagnostic only.
                    lc_args, lc_kwargs, lc_out = tke_capture["lc_calls"][0]
                    if lc_kwargs:
                        ice_lc = lc_kwargs.get("ice_frac")
                        if ice_lc is not None and np.any(np.asarray(ice_lc)):
                            raise AssertionError(
                                "row 10 preregisters DINO's no-ice branch")
                    taum_l, n2_l, depth_l, dz_l, lc_cfg = lc_args[:5]
                    nlc = lc_out.shape[-1]

                    def _as_lc_3d(value):
                        arr = np.asarray(value)
                        return np.broadcast_to(
                            arr, (wet2.shape[0], wet2.shape[1], nlc))

                    n2_l_np = _as_lc_3d(n2_l)
                    depth_l_np = _as_lc_3d(depth_l)
                    dz_l_np = _as_lc_3d(dz_l)
                    with xr.open_dataset(
                            RUN / "mesh_mask.nc", decode_times=False) as ds:
                        gdepw0_lc = np.moveaxis(np.asarray(
                            ds["gdepw_0"].isel(time_counter=0)), 0, -1)
                        e3w0_lc = np.moveaxis(np.asarray(
                            ds["e3w_0"].isel(time_counter=0)), 0, -1)
                    stretch_lc = np.asarray(nemo_r3t_stretch(
                        br.z_coord, twin_state.eta.data,
                        twin_state.H_bathy.data))
                    depth_n_lc = (gdepw0_lc[..., 1:1 + nlc]
                                  * stretch_lc[..., None])
                    dz_n_lc = (e3w0_lc[..., 1:1 + nlc]
                               * stretch_lc[..., None])
                    n2_n_lc = n2_n_full[..., 1:1 + nlc]
                    wet_lc = wet_w_all[..., :nlc]

                    zcsd = (np.float64(0.5) * np.float64(0.016)
                            * np.float64(0.016)
                            / (np.float64(1.22) * np.float64(1.5e-3)))
                    half_n = zcsd * taum_dump
                    pe_n = np.cumsum(
                        np.maximum(n2_n_lc, 0.0) * depth_n_lc * dz_n_lc,
                        axis=-1)
                    exceeded_n = pe_n > half_n[..., None]
                    first_n = np.argmax(exceeded_n, axis=-1)
                    fallback_n = np.clip(nemo_bottom_level, 0, nlc - 1)
                    imlc_n = np.where(
                        np.any(exceeded_n, axis=-1), first_n, fallback_n)
                    hlc_n = np.take_along_axis(
                        depth_n_lc, imlc_n[..., None], axis=-1)[..., 0]
                    us_n = np.sqrt(np.float64(2.0) * half_n)
                    us3_n = us_n * us_n * us_n
                    zwlc_n = (np.float64(lc_cfg.lc_coeff)
                              * np.sin(np.float64(np.pi)
                                       * depth_n_lc / hlc_n[..., None]))
                    src_n = (us3_n[..., None]
                             * (zwlc_n * zwlc_n * zwlc_n)
                             / hlc_n[..., None])
                    src_n = np.where(
                        (depth_n_lc - hlc_n[..., None] < 0.0) & wet_lc,
                        src_n, 0.0)

                    row10m = metrics(
                        np.asarray(lc_out), src_n, wet_lc, focus,
                        POINTWISE_BAR)
                    row10_inputs = {
                        "taum": metrics(
                            np.asarray(taum_l)[..., None],
                            taum_dump[..., None], wet2[..., None], focus,
                            POINTWISE_BAR),
                        "rn2b": metrics(
                            n2_l_np, n2_n_lc, wet_lc, focus, POINTWISE_BAR),
                        "gdepw_Kmm": metrics(
                            depth_l_np, depth_n_lc, wet_lc, focus,
                            POINTWISE_BAR),
                        "e3w_Kmm": metrics(
                            dz_l_np, dz_n_lc, wet_lc, focus, POINTWISE_BAR),
                        "rn2b_vs_verified_row2_production": metrics(
                            n2_l_np, np.asarray(n2)[..., :nlc], wet_lc,
                            focus, POINTWISE_BAR),
                        "gdepw_vs_row2_live_geometry": metrics(
                            depth_l_np, np.asarray(gdepw)[..., :nlc], wet_lc,
                            focus, POINTWISE_BAR),
                        "e3w_vs_row2_live_geometry": metrics(
                            dz_l_np, np.asarray(e3w)[..., :nlc], wet_lc,
                            focus, POINTWISE_BAR),
                        "rn2b_offset0": metrics(
                            n2_l_np, n2_n_full[..., :nlc], wet_lc,
                            focus, POINTWISE_BAR),
                        "gdepw_offset0": metrics(
                            depth_l_np,
                            gdepw0_lc[..., :nlc] * stretch_lc[..., None],
                            wet_lc, focus, POINTWISE_BAR),
                        "e3w_offset0": metrics(
                            dz_l_np,
                            e3w0_lc[..., :nlc] * stretch_lc[..., None],
                            wet_lc, focus, POINTWISE_BAR),
                    }
                    # The independently dumped rn2b is the first source-order
                    # operand, so its comparison -- not the undumped offline
                    # composite -- owns the row disposition.
                    row10_operand = row10_inputs["rn2b"]
                    row10_operand["controls"] = planted_controls(
                        np.asarray(n2)[..., :nlc], n2_n_lc, wet_lc,
                        POINTWISE_BAR)
                    row10_operand["offline_composite_diagnostic"] = {
                        "dispositive": False,
                        "reason": "no NEMO post-ln_lc stage dump",
                        "metrics": row10m,
                    }
                    row4["continuation_preview"]["rows"][
                        "10_langmuir_rn2b_operand"] = row10_operand
                    if not row10_operand["pass"]:
                        row4["continuation_preview"]["first_divergence"] = {
                            "row": 10,
                            "operation": "Langmuir TKE source rn2b operand",
                            "nemo_line":
                                "cfgs/DINO/MY_SRC/zdftke.F90:436-440",
                            "output": row10_operand,
                            "first_failing_operand": {
                                "name": "rn2b evaluation lifetime/geometry",
                                "nemo_line":
                                    "cfgs/DINO/MY_SRC/stpmlf.F90:204-210; "
                                    "cfgs/DINO/MY_SRC/zdftke.F90:436-440",
                                "legoesm_line":
                                    "packages/ocean/legoesm/ocean/physics/"
                                    "vertical_mixing/k_profiles.py:827-831",
                            },
                            "operand_localization": row10_inputs,
                            "offline_composite_diagnostic": {
                                "dispositive": False,
                                "reason": "no NEMO post-ln_lc stage dump",
                                "metrics": row10m,
                            },
                            "next_round_fix_design": {
                                "option": "tke_n2_evaluation_stage",
                                "faithful_default_on_complete_dino_nemo_cards":
                                    "step_entry",
                                "legacy_default_everywhere_else":
                                    "implicit_solve_state",
                                "legacy_opt_in_on_dino_nemo_cards":
                                    "implicit_solve_state",
                                "construction": "at model-step entry compute "
                                    "and freeze the exact (rn2, rn2b, "
                                    "gdepw_Kmm, e3w_Kmm) bundle from the "
                                    "registered raw-mesh live geometry; carry "
                                    "it through zdf_mxl and every zdf_tke "
                                    "consumer. Langmuir must receive frozen "
                                    "gdepw_Kmm/e3w_Kmm rather than "
                                    "-z_interface/dz_half. Non-oracle cards "
                                    "retain the present recomputation and "
                                    "metric slots byte-for-byte.",
                                "required_source_order_after_fix": [
                                    "zWlc2", "zpelc", "imlc", "zhlc",
                                    "zus3", "zwlc", "source"],
                            },
                            "controls": planted_controls(
                                src_n, src_n, wet_lc, POINTWISE_BAR),
                        }
        if row4["disposition"] == "DIVERGED":
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel,
            )
            cfg_legacy = tke_cfg._replace(
                tke_shear_avm_weighting="tpoint",
                tke_shear_evaluation_stage="implicit_solve_state")
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
                    e3u_0 = np.moveaxis(np.asarray(
                        ds["e3u_0"].isel(time_counter=0)), 0, -1)
                    e3v_0 = np.moveaxis(np.asarray(
                        ds["e3v_0"].isel(time_counter=0)), 0, -1)
                    e3uw_0 = np.moveaxis(np.asarray(
                        ds["e3uw_0"].isel(time_counter=0)), 0, -1)
                    e3vw_0 = np.moveaxis(np.asarray(
                        ds["e3vw_0"].isel(time_counter=0)), 0, -1)
                    e1t = np.asarray(ds["e1t"].isel(time_counter=0)).squeeze()
                    e2t = np.asarray(ds["e2t"].isel(time_counter=0)).squeeze()
                    e1u = np.asarray(ds["e1u"].isel(time_counter=0)).squeeze()
                    e2u = np.asarray(ds["e2u"].isel(time_counter=0)).squeeze()
                    e1v = np.asarray(ds["e1v"].isel(time_counter=0)).squeeze()
                    e2v = np.asarray(ds["e2v"].isel(time_counter=0)).squeeze()
                u_now_arg, v_now_arg, u_b_arg, v_b_arg = sh2_args[:4]
                u_now = np.asarray(u_now_arg)[:, 1:, :]
                v_now = np.asarray(v_now_arg)[1:, :, :]
                u_before = np.asarray(u_b_arg)[:, 1:, :]
                v_before = np.asarray(v_b_arg)[1:, :, :]
                velocity_differences = {
                    "u_now": face_difference_metrics(
                        u_now, now.u, umask, focus, "u"),
                    "v_now": face_difference_metrics(
                        v_now, now.v, vmask, focus, "v"),
                    "u_before": face_difference_metrics(
                        u_before, before.u, umask, focus, "u"),
                    "v_before": face_difference_metrics(
                        v_before, before.v, vmask, focus, "v"),
                }
                now_sub = sh2_fn(
                    twin_state.u.data, twin_state.v.data,
                    *sh2_args[2:], **sh2_kwargs)
                now_sub_m = metrics(
                    np.asarray(now_sub)[..., :nsh], sh2_n, wet_sh2,
                    focus, POINTWISE_BAR)

                # Continue after the now-velocity fix in literal line order:
                # NOW metric, BEFORE metric, masks, then four-face assembly.
                hu_0 = (e3u_0 * umask).sum(axis=-1)
                hv_0 = (e3v_0 * vmask).sum(axis=-1)
                ssumask = (umask.max(axis=-1) > 0).astype(np.float64)
                ssvmask = (vmask.max(axis=-1) > 0).astype(np.float64)
                r3u_n, r3v_n = sh2_walk.qco_r3(
                    now.ssh, e1t, e2t, e1u, e2u, e1v, e2v,
                    hu_0, hv_0, ssumask, ssvmask)
                r3u_b, r3v_b = sh2_walk.qco_r3(
                    before.ssh, e1t, e2t, e1u, e2u, e1v, e2v,
                    hu_0, hv_0, ssumask, ssvmask)

                def _u_raw(metric):
                    metric = metric[..., 1:]
                    return jnp.asarray(np.concatenate(
                        [metric[:, -1:, :], metric], axis=1))

                def _v_raw(metric):
                    metric = metric[..., 1:]
                    return jnp.asarray(np.concatenate(
                        [metric[:1, :, :], metric], axis=0))

                e3un = _u_raw(e3uw_0 * (1.0 + r3u_n[..., None]))
                e3ub = _u_raw(e3uw_0 * (1.0 + r3u_b[..., None]))
                e3vn = _v_raw(e3vw_0 * (1.0 + r3v_n[..., None]))
                e3vb = _v_raw(e3vw_0 * (1.0 + r3v_b[..., None]))
                dz = sh2_args[4]
                e3u_legacy = jnp.concatenate([dz, dz[:, -1:, :]], axis=1)
                e3v_legacy = jnp.concatenate([dz, dz[-1:, :, :]], axis=0)
                u_mask_arg, v_mask_arg = sh2_args[5:7]
                avm_arg = sh2_args[-1]

                def _score_candidate(eun, eub, evn, evb, um, vm):
                    value = assemble_sh2_with_metrics(
                        twin_state.u.data, twin_state.v.data,
                        sh2_args[2], sh2_args[3], avm_arg,
                        eun, eub, evn, evb, um, vm)
                    return metrics(np.asarray(value)[..., :nsh], sh2_n,
                                   wet_sh2, focus, POINTWISE_BAR)

                metric_candidates = []
                now_metric_m = _score_candidate(
                    e3un, e3u_legacy, e3vn, e3v_legacy,
                    u_mask_arg, v_mask_arg)
                metric_candidates.append({
                    "subrow": "4c", "operand": "live NOW e3uw/e3vw(Kmm)",
                    "nemo_line": "cfgs/DINO/WORK/zdfsh2.F90:83,88",
                    "metrics": now_metric_m})
                both_metric_m = _score_candidate(
                    e3un, e3ub, e3vn, e3vb, u_mask_arg, v_mask_arg)
                metric_candidates.append({
                    "subrow": "4d", "operand": "live BEFORE e3uw/e3vw(Kbb)",
                    "nemo_line": "cfgs/DINO/WORK/zdfsh2.F90:83,88",
                    "metrics": both_metric_m})

                u_mask_exact = jnp.asarray(np.concatenate(
                    [umask[:, -1:, :], umask], axis=1))
                v_mask_exact = jnp.asarray(np.concatenate(
                    [vmask[:1, :, :], vmask], axis=0))
                exact_mask_m = _score_candidate(
                    e3un, e3ub, e3vn, e3vb,
                    u_mask_exact, v_mask_exact)
                metric_candidates.append({
                    "subrow": "4e", "operand": "wumask/wvmask",
                    "nemo_line": "cfgs/DINO/WORK/zdfsh2.F90:84,89",
                    "metrics": exact_mask_m})

                exact_nemo = sh2_walk.zdf_sh2_reconstruct(
                    now.u, now.v, before.u, before.v,
                    avm_n_full,
                    e3uw_0 * (1.0 + r3u_n[..., None]),
                    e3uw_0 * (1.0 + r3u_b[..., None]),
                    e3vw_0 * (1.0 + r3v_n[..., None]),
                    e3vw_0 * (1.0 + r3v_b[..., None]),
                    umask, vmask)
                exact_assembly_m = metrics(
                    exact_nemo[..., 1:1 + nsh], sh2_n, wet_sh2,
                    focus, POINTWISE_BAR)
                metric_candidates.append({
                    "subrow": "4f", "operand": "four-face/coastal assembly",
                    "nemo_line": "cfgs/DINO/WORK/zdfsh2.F90:92-94",
                    "metrics": exact_assembly_m})
                next_operand = {
                    "name": "NOW face-velocity differences at zdf_phy entry",
                    "nemo_line": "cfgs/DINO/WORK/zdfsh2.F90:81-82,86-87",
                    "legoesm_operand": "post-explicit/pre-implicit-solve u/v",
                    "velocity_difference_operands": velocity_differences,
                    "substitute_step_entry_now_velocities": now_sub_m,
                    "remaining_operand_substitutions": metric_candidates,
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
                "production_composite": row4m,
                "composite_with_verified_carried_avm": avm_sub_m,
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
        Path(__file__).resolve(),
        HERE / "PREREG_zdf_chain_sweep.md",
        HERE / "PREREG_zdf_chain_sweep_round3.md",
        HERE / "PREREG_zdf_chain_sweep_round4.md",
        HERE / "kamm_twin_90d.py",
        Path("packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
        Path("packages/ocean/legoesm/ocean/eos.py"),
        Path("packages/ocean/legoesm/ocean/experiments/dino.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/nemo_io.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/time_levels.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py"),
        Path("packages/ocean/legoesm/ocean/vertical.py"),
        Path("packages/ocean/legoesm/ocean/state.py"),
        HERE / "dump_lane.py",
        HERE / "eos_rab_bn2_per_element.py",
        ORACLE / "cfgs/DINO/MY_SRC/stpmlf.F90",
        ORACLE / "cfgs/DINO/MY_SRC/usrdef_sbc.F90",
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
        RUN / "tke_dump_dissl.bin",
        RUN / "tke_dump_pdlr.bin",
        RUN / "tke_dump_en.bin",
        RUN / "dump_nmln.bin",
        RUN / "dump_hmlp.bin",
        RUN / "drg_dump_rCdU_bot.bin",
        RUN / "sbc_dump_utau.bin",
        RUN / "eiv_dump_gdept.bin", RUN / "eiv_dump_e3w.bin", args.mld_maps,
    ]
    continuation = row4.get("continuation_preview")
    rows = {"1_eos_rab_before": row1, "2_bn2_before": row2,
            "3_eos_rab_bn2_now": row3, "4_zdf_sh2": row4}
    if continuation is not None:
        for name, value in continuation["rows"].items():
            rows[name] = {
                "output": value,
                "disposition": "VERIFIED" if value["pass"] else "DIVERGED",
            }
    elif row4["disposition"] == "DIVERGED":
        rows["5_bottom_drag_coefficient"] = {
            "disposition": "UNMEASURED", "reason": "stop at row 4"}

    if row2["disposition"] == "DIVERGED":
        first_divergence = {
            "row": 2, "operation": "bn2(Nbb)",
            "nemo_line": "src/OCE/TRA/eosbn2.F90:1467",
            "localization": localization,
        }
    elif row3["disposition"] == "DIVERGED":
        first_divergence = {
            "row": 3, "operation": "eos_rab/bn2(Nnn)",
            "nemo_line": "src/OCE/TRA/eosbn2.F90:1467",
            "localization": None,
        }
    elif row4["disposition"] == "DIVERGED":
        first_divergence = {
            "row": 4, "operation": "zdf_sh2",
            "nemo_line": "src/OCE/ZDF/zdfsh2.F90:80-94",
            "localization": sh2_localization,
        }
    elif continuation is not None:
        first_divergence = continuation.get("first_divergence")
    else:
        first_divergence = None
    artifact = {
        "schema": "zdf-chain-sweep-v6",
        "lane": "d180", "kt": 5761, "cpu_only": True, "fp64": True,
        "checked_out_parent_sha": git_sha(),
        "probe_commit_sha": probe_commit_sha(),
        "effective_env": {
            "DINO_1226_LANE": os.environ["DINO_1226_LANE"],
            **{name: os.environ[name] for name in _ENV_DEFAULTS},
            "rejected_ablation_overrides": [
                "DINO_EEN_METRIC", "DINO_BOLUS_ADV",
                "DINO_RECONCILE_TARGET", "DINO_AFTER_RECONCILE",
                "DINO_NEMO_KMM_DIVISOR", "DINO_OUTER_INTEGRATOR",
                "DINO_TWIN_SEASONAL_KT0",
            ],
        },
        "platform": platform.platform(), "python": platform.python_version(),
        "jax": importlib.metadata.version("jax"), "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "time_levels": {x: time_level_for_dump(x) for x in
                        ("dump_alpha_b.bin", "dump_beta_b.bin", "tke_dump_rn2b.bin",
                         "tke_dump_rn2.bin",
                         "tke_dump_sh2.bin",
                         "tke_dump_avm_in.bin",
                         "tke_dump_en.bin",
                         "dump_nmln.bin", "dump_hmlp.bin",
                         "drg_dump_rCdU_bot.bin", "sbc_dump_utau.bin",
                         "eiv_dump_gdept.bin", "eiv_dump_e3w.bin")},
        "focus_columns_ji": [list(x) for x in focus],
        "bars": {"pointwise_column": POINTWISE_BAR, "corr": CORR_BAR,
                 "rms_ratio_epsilon": RATIO_EPS},
        "controls": controls,
        "rows": rows,
        "first_divergence": first_divergence,
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
            am = sh2_localization["fixed_operand"]["metrics"]
            print(f"  fixed carried p_avm operand: pass={am['pass']} "
                  f"max={am['max_column_error']:.6e} "
                  f"bad_columns={am['n_diverged_columns']}/{am['n_wet_columns']}")
            nxt = sh2_localization["first_diverging_operand"]
            if nxt is not None:
                u = nxt["velocity_difference_operands"]["u_now"]
                v = nxt["velocity_difference_operands"]["v_now"]
                sm = nxt["substitute_step_entry_now_velocities"]
                print(f"  next operand NOW u: pass={u['pass']} "
                      f"max={u['max_column_error']:.6e} "
                      f"bad_columns={u['n_diverged_columns']}/{u['n_wet_columns']}")
                print(f"  next operand NOW v: pass={v['pass']} "
                      f"max={v['max_column_error']:.6e} "
                      f"bad_columns={v['n_diverged_columns']}/{v['n_wet_columns']}")
                print(f"  substitute step-entry NOW velocities: pass={sm['pass']} "
                      f"max={sm['max_column_error']:.6e} "
                      f"bad_columns={sm['n_diverged_columns']}/{sm['n_wet_columns']}")
                for candidate in nxt.get("remaining_operand_substitutions", []):
                    cm = candidate["metrics"]
                    print(f"  {candidate['subrow']} substitute {candidate['operand']}: "
                          f"pass={cm['pass']} max={cm['max_column_error']:.6e} "
                          f"bad_columns={cm['n_diverged_columns']}/{cm['n_wet_columns']}")
        preview = row4.get("continuation_preview")
        if preview is not None:
            for name, cm in preview["rows"].items():
                print(f"row {name}: {'VERIFIED' if cm['pass'] else 'DIVERGED'} "
                      f"max={cm['max_column_error']:.6e} "
                      f"bad_columns={cm['n_diverged_columns']}/{cm['n_wet_columns']}")
            first = preview.get("first_divergence")
            if first is not None:
                print(f"  first divergence row {first['row']}: "
                      f"{first['operation']}")
                for name, cm in first["operand_localization"].items():
                    if isinstance(cm, dict) and "pass" in cm:
                        print(f"    {name}: pass={cm['pass']} "
                              f"max={cm['max_column_error']:.6e} "
                              f"bad_columns={cm['n_diverged_columns']}/"
                              f"{cm['n_wet_columns']}")
    if localization:
        for c in localization["candidates_in_nemo_evaluation_order"]:
            print(f"  substitute {c['substitution']}: pass={c['metrics']['pass']} "
                  f"max={c['metrics']['max_column_error']:.6e}")
        print(f"  first_passing_substitution={localization['first_passing_substitution']}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
