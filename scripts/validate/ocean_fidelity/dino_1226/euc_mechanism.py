#!/usr/bin/env python
"""Day-180 matched-state EUC vertical-mixing mechanism probe.

This probe scores legoESM's shipped TKE closure against NEMO's instrumented
``avm``/``avt`` fields on the *same* day-180 state.  It deliberately does not
infer closure coefficients from restart ``avm_k``: the selected D180 dump lane
contains the coefficients produced at the first matched step and therefore
removes the restart-time-level ambiguity.

The registered short-run arm is described by :func:`substitution_design`.
Running it requires a visible CUDA device; this offline scorer remains useful
and fail-closed without one.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import netCDF4 as nc
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

# Hash pins are receipts for reused campaign logic, not documentation-only
# hashes.  Refuse to run if a sibling changes beneath this probe.
PINNED = {
    "kamm_twin_90d.py": "00b5f0d6c6c58ef984e010825239af27f768b83cf431e8763d236325c7cf68e6",
    "bn2_alpha_compare.py": "2ac9e4a457c4dded55a116d67938dfda035e79a3a98ccbb070d9a0ec3d11e40e",
    "dump_lane.py": "41252d898b0d09b350c086aa678ddb890c7e6232fa2b46e88093a3b108a9bf62",
    "zdftke_avm_offset_scan.py": "7ec40312dba6d70b7d6bb9ed5349b5861c7fa1b1ab4859b7c2f9c5c785300feb",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_pins() -> dict[str, str]:
    observed = {name: sha256(HERE / name) for name in PINNED}
    bad = {n: (PINNED[n], observed[n]) for n in PINNED if observed[n] != PINNED[n]}
    if bad:
        raise SystemExit(f"reused reducer/loader hash mismatch: {bad}")
    return observed


def import_sibling(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_euc_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_receipt(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {
            "head": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "branch": None, "dirty": None}


def weighted_nrms(lego: np.ndarray, nemo: np.ndarray, weights: np.ndarray,
                  wet: np.ndarray) -> float:
    """Thickness-weighted RMS difference normalized by NEMO RMS."""
    valid = wet & np.isfinite(lego) & np.isfinite(nemo) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("empty wet comparison")
    w = weights[valid]
    num = np.sum(w * np.square(lego[valid] - nemo[valid]))
    den = np.sum(w * np.square(nemo[valid]))
    if den <= 0:
        raise ValueError("zero NEMO weighted norm")
    return float(np.sqrt(num / den))


def weighted_level_mean(a: np.ndarray, weights: np.ndarray,
                        wet: np.ndarray) -> np.ndarray:
    valid = wet & np.isfinite(a) & np.isfinite(weights) & (weights > 0)
    den = np.sum(np.where(valid, weights, 0.0), axis=0)
    num = np.sum(np.where(valid, weights * a, 0.0), axis=0)
    return np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)


def classify_offline(nrms: float, ratios: np.ndarray) -> str:
    finite = ratios[np.isfinite(ratios)]
    runs = 0
    last_sign = 0
    for r in finite:
        sign = -1 if r < 0.75 else (1 if r > 1.25 else 0)
        runs = runs + 1 if sign and sign == last_sign else (1 if sign else 0)
        last_sign = sign
        if runs >= 2 and nrms >= 0.25:
            return "CONFIRM_CLOSURE_DIFFERENCE"
    if nrms <= 0.10 and np.all((finite >= 0.90) & (finite <= 1.10)):
        return "REFUTE_CLOSURE_DIFFERENCE"
    return "UNRESOLVED_CLOSURE_DIFFERENCE"


def classify_offline_review(nrms: float, ratios: np.ndarray,
                            depths_m: np.ndarray) -> str:
    """A3/A4 review reissue: a shear-setting surface level may confirm alone."""
    finite = np.isfinite(ratios)
    if nrms <= 0.10 and np.all((ratios[finite] >= 0.90) & (ratios[finite] <= 1.10)):
        return "REFUTE_VISCOSITY_PRIME_SUSPECT"
    consecutive = False
    last = 0
    run = 0
    for r in ratios[finite]:
        sign = -1 if r < 0.75 else (1 if r > 1.25 else 0)
        run = run + 1 if sign and sign == last else (1 if sign else 0)
        last = sign
        consecutive |= run >= 2
    shear_setting = np.flatnonzero(finite & (depths_m >= 5.0) & (depths_m <= 26.0))
    # Excess viscosity is the registered sign that predicts weaker shear and
    # a deeper zero crossing; a deficit would not predict the observed pair.
    single_predictive = bool(np.any(ratios[shear_setting] > 1.25))
    if nrms >= 0.25 and (consecutive or single_predictive):
        return "CONFIRM_VISCOSITY_PRIME_SUSPECT"
    return "UNRESOLVED_VISCOSITY_PRIME_SUSPECT"


def closure_attribution(lego_avm: np.ndarray, nemo_avm: np.ndarray,
                        lego_en: np.ndarray, nemo_en: np.ndarray,
                        lego_mxl: np.ndarray, nemo_mxl: np.ndarray,
                        wet: np.ndarray, *, c_lego: float, c_nemo: float,
                        floor_lego: float, floor_nemo: float) -> dict[str, Any]:
    """Registered per-column log-factor attribution at the 10.14 m interface."""
    arrays = (lego_avm, nemo_avm, lego_en, nemo_en, lego_mxl, nemo_mxl)
    finite = wet.copy()
    for a in arrays:
        finite &= np.isfinite(a) & (a > 0)
    eligible = finite & (lego_avm > floor_lego * (1.0 + 1e-12)) \
                      & (nemo_avm > floor_nemo * (1.0 + 1e-12))
    n_wet = int(np.count_nonzero(wet))
    n_eligible = int(np.count_nonzero(eligible))
    if n_eligible < 0.75 * n_wet:
        raise ValueError(f"attribution eligibility {n_eligible}/{n_wet} is below 75%")

    recon_l = c_lego * lego_mxl * np.sqrt(lego_en)
    recon_n = c_nemo * nemo_mxl * np.sqrt(nemo_en)
    rel_l = np.abs(recon_l[eligible] / lego_avm[eligible] - 1.0)
    rel_n = np.abs(recon_n[eligible] / nemo_avm[eligible] - 1.0)
    max_recon = float(max(np.max(rel_l), np.max(rel_n)))
    if max_recon > 1e-10:
        raise ValueError(f"three-factor avm reconstruction failed: {max_recon:.3e}")

    d_c = np.full(lego_avm.shape, np.log(c_lego / c_nemo))
    d_l = np.log(lego_mxl / nemo_mxl)
    d_e = 0.5 * np.log(lego_en / nemo_en)
    d_a = np.log(lego_avm / nemo_avm)
    closure = float(np.max(np.abs((d_c + d_l + d_e - d_a)[eligible])))
    if closure > 1e-10:
        raise ValueError(f"log-factor closure failed: {closure:.3e}")
    excess = eligible & (lego_avm / nemo_avm > 1.25)
    if not np.any(excess):
        raise ValueError("no >1.25 excess columns for registered attribution")

    pieces = {"coefficient_stability": d_c, "mixing_length": d_l, "tke_energy": d_e}
    scores = {name: float(np.median(np.abs(v[excess]))) for name, v in pieces.items()}
    score_sum = sum(scores.values())
    detail = {}
    carriers = []
    for name, values in pieces.items():
        vals = values[excess]
        share = scores[name] / score_sum if score_sum else 0.0
        agreement = float(np.mean(np.sign(vals) == np.sign(d_a[excess])))
        detail[name] = {
            "median_abs_log_score": scores[name],
            "score_share": share,
            "sign_agreement_fraction": agreement,
            "geometric_mean_factor": float(np.exp(np.mean(vals))),
            "factor_iqr": np.exp(np.quantile(vals, [0.25, 0.75])).tolist(),
        }
        if share >= 0.60 and agreement >= 0.75:
            carriers.append(name)
    label = (f"{carriers[0].upper()}_CARRIES_10M_EXCESS"
             if len(carriers) == 1 else "DISTRIBUTED_OR_UNRESOLVED")
    return {
        "label": label,
        "n_wet": n_wet,
        "n_eligible": n_eligible,
        "eligible_fraction": n_eligible / n_wet,
        "n_excess": int(np.count_nonzero(excess)),
        "max_direct_reconstruction_relative_error": max_recon,
        "max_log_factor_closure_error": closure,
        "direct_zonal_mean_ratio": float(np.mean(lego_avm[wet]) / np.mean(nemo_avm[wet])),
        "pieces": detail,
        "executed_branch": (
            "MY_SRC/zdftke.F90:832-837: zsqen=SQRT(en); "
            "zav=rn_ediff*zmxlm*zsqen; p_avm=MAX(zav,avmb)*wmask. "
            "No pdlr/stability multiplier acts on avm; :841-844 updates avt only."
        ),
    }


def wrong_shift(a: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full_like(a, fill)
    out[..., :-1] = a[..., 1:]
    return out


def substitution_design() -> dict[str, Any]:
    return {
        "duration_days": 10,
        "steps": 320,
        "device": "CUDA_VISIBLE_DEVICES=0",
        "initial_state": "bit-identical NEMO/legoESM day-180 now,before,TKE restart bridge",
        "base": "shipped nemo_dino_kamm_mlf",
        "arm": (
            "at every legoESM step replace only non-EVD momentum viscosity K_m "
            "passed to the implicit vertical momentum solve by frozen NEMO day-180 "
            "tke_dump_avm_final[jk=2..36]; retain legoESM K_v, TKE evolution, "
            "surface forcing, and cells where the native EVD trigger sets K_m=100 m2/s"
        ),
        "readout": "daily equatorial wet-zonal-mean u(z); shallowest 0-100 m zero crossing and 2-level shear",
        "confirm": "Cz >= 0.50 toward NEMO and shear absolute error worsens by no more than 10%",
        "refute": "Cz <= 0.10 or response is away from NEMO",
        "unmeasurable": "BASE day-10 core-depth error < 0.5 m",
        "controls": [
            "day-0 BASE/arm state identity bit-for-bit",
            "hook returns bit-identical K_v and TKE to BASE and changes K_m only outside native EVD cells",
            "dry/wet-mask plant and one-level vertical-shift plant must fail",
            "2x K_m plant must change first-step momentum tendency",
            "BASE must reproduce the archived verdict360 control at day 10",
            "read and re-record every NPZ provenance stamp and all coefficient/source hashes",
        ],
        "cost": {
            "gpu_seconds_per_arm_estimate_including_jit": 93,
            "gpu_seconds_pair_estimate_including_separate_jit": 186,
            "steady_state_gpu_seconds_per_10_day_arm": 18,
            "basis": (
                "archived shipped m0_control.log reports day 10 wall=93 s and "
                "day 360 wall=637 s on one GPU; separate BASE/arm processes each compile"
            ),
            "storage_estimate": "roughly 25 MiB per arm with only day-0/day-10 3-D snapshots",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dino_euc_mechanism/euc_mechanism.json")
    args = ap.parse_args()
    pins = require_pins()

    # Imports happen only after pins pass.
    kt = import_sibling("kamm_twin_90d")
    bac = import_sibling("bn2_alpha_compare")
    dl = import_sibling("dump_lane")
    zos = import_sibling("zdftke_avm_offset_scan")
    if dl.LANE != "d180":
        raise SystemExit(f"expected DINO_1226_LANE=d180, got {dl.LANE!r}")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.precision_gate import require_fp64
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    import legoesm.ocean.physics.vertical_mixing.tke as tke_module
    set_policy(PrecisionPolicy.fp64())

    br, cfg, mc, model, forcing, sf, state = kt._build_twin_state(
        "nemo_dino_kamm_mlf", dl.RUN_DIR, dl.RUN_DIR,
        bridge_tke=True, bridge_before=True, restart_file=dl.RESTART,
        e3t_mode="both")
    require_fp64(state, context="EUC day-180 matched-state probe")
    module_path = Path(importlib.util.find_spec("legoesm.ocean.physics.vertical_mixing").origin).resolve()
    if ROOT not in module_path.parents:
        raise SystemExit(f"vertical mixing imported outside this worktree: {module_path}")

    K_prod, A_prod = zos.run_and_capture(model, state, sf)
    conv_off = mc.physics.convection._replace(scheme="none")
    closure_model = LatLonCGridOceanModel(
        br.geometry, br.z_coord,
        mc._replace(physics=mc.physics._replace(convection=conv_off)))
    # Capture the actual final e and mixing length passed to the production
    # coefficient constructor. This is observation of the shipped path, not a
    # parallel reimplementation of the TKE closure.
    real_compute_k = tke_module.compute_K_from_tke
    component_calls: list[dict[str, Any]] = []

    def component_spy(e, l_k, tke_cfg, *a, **kw):
        out = real_compute_k(e, l_k, tke_cfg, *a, **kw)
        component_calls.append({
            "e": np.asarray(e), "l_k": np.asarray(l_k),
            "K_M": np.asarray(out[0]), "c_k": float(tke_cfg.c_k),
            "kappa_convention": str(tke_cfg.kappa_convention),
            "floor": float(tke_cfg.kappaM_min),
        })
        return out

    tke_module.compute_K_from_tke = component_spy
    try:
        K_cl, A_cl = zos.run_and_capture(closure_model, state, sf)
    finally:
        tke_module.compute_K_from_tke = real_compute_k
    if A_prod is None or A_cl is None:
        raise SystemExit("momentum viscosity was not returned")
    if not component_calls:
        raise SystemExit("TKE component spy captured no coefficient calls")
    shape_calls = [c for c in component_calls if c["K_M"].shape == A_cl.shape]
    if not shape_calls:
        raise SystemExit("no captured TKE coefficient has the closure output shape")
    component_final = min(
        shape_calls, key=lambda c: float(np.max(np.abs(c["K_M"] - A_cl))))
    component_output_max_abs = float(np.max(np.abs(component_final["K_M"] - A_cl)))

    jpi, jpj, jpk, hls = bac._read_dims(dl.RUN_DIR)
    avm = bac._load_interior(dl.dump_path("tke_dump_avm_final.bin"), jpi - 2*hls, jpj - 2*hls)
    avt = bac._load_interior(dl.dump_path("tke_dump_avt_final.bin"), jpi - 2*hls, jpj - 2*hls)
    en_nemo = bac._load_interior(dl.dump_path("tke_dump_en.bin"), jpi - 2*hls, jpj - 2*hls)
    mxl_nemo = bac._load_interior(dl.dump_path("tke_dump_zmxlm.bin"), jpi - 2*hls, jpj - 2*hls)
    component_time_levels = {
        name: time_level_for_dump(name) for name in (
            "tke_dump_en.bin", "tke_dump_zmxlm.bin",
            "tke_dump_avm_final.bin", "dump_avm.bin", "dump_avt.bin")
    }
    # zdf_phy's post-EVD dumps retain the model halos; zdftke's internal
    # final-coefficient dumps above are already interior-only.
    avm_real = bac._load_haloed(dl.dump_path("dump_avm.bin"), jpi, jpj, hls)
    avt_real = bac._load_haloed(dl.dump_path("dump_avt.bin"), jpi, jpj, hls)

    with nc.Dataset(Path(dl.RUN_DIR) / "mesh_mask.nc") as mm:
        def llz(name: str) -> np.ndarray:
            return np.moveaxis(np.asarray(mm[name][0]).squeeze(), 0, -1)
        tmask = llz("tmask") > 0.5
        e3t = llz("e3t_0")
        gphit = np.asarray(mm["gphit"][0]).squeeze()
        gdepw = np.asarray(mm["gdepw_1d"][:]).squeeze()
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]
    e3w = 0.5 * (e3t[..., :-1] + e3t[..., 1:])

    equator_row = int(np.nanargmin(np.abs(np.nanmean(gphit, axis=1))))
    equator_lat = float(np.nanmean(gphit[equator_row]))
    # The post-EVD dump stops at jk=35, so its common interior set is 34
    # interfaces (jk=2..35).  Use that common set for all four comparisons;
    # the scored 10--40 m window is near the surface and loses no levels.
    nk = min(A_cl.shape[-1], avm.shape[-1] - 1,
             avm_real.shape[-1] - 1, e3w.shape[-1])
    z = gdepw[1:1+nk]
    score_k = (z >= 10.0) & (z <= 40.0)
    wet = wmask[equator_row, :, 1:1+nk]
    weights = e3w[equator_row, :, :nk]

    pairs = {
        "closure_avm": (A_cl[equator_row, :, :nk], avm[equator_row, :, 1:1+nk]),
        "closure_avt": (K_cl[equator_row, :, :nk], avt[equator_row, :, 1:1+nk]),
        "realized_avm": (A_prod[equator_row, :, :nk], avm_real[equator_row, :, 1:1+nk]),
        "realized_avt": (K_prod[equator_row, :, :nk], avt_real[equator_row, :, 1:1+nk]),
    }
    scored: dict[str, Any] = {}
    for name, (lego, nemo) in pairs.items():
        wm = wet[:, score_k]
        ww = weights[:, score_k]
        ll = lego[:, score_k]
        nn = nemo[:, score_k]
        lm = weighted_level_mean(ll, ww, wm)
        nm = weighted_level_mean(nn, ww, wm)
        ratios = np.divide(lm, nm, out=np.full_like(lm, np.nan), where=np.abs(nm) > 0)
        nrms = weighted_nrms(ll, nn, ww, wm)
        core20 = z[score_k] >= 20.0
        core26 = z[score_k] >= 26.0
        scored[name] = {
            "nrms": nrms,
            "diagnostic_nrms_20_40m": weighted_nrms(
                ll[:, core20], nn[:, core20], ww[:, core20], wm[:, core20]),
            "diagnostic_nrms_26_40m": weighted_nrms(
                ll[:, core26], nn[:, core26], ww[:, core26], wm[:, core26]),
            "level_depth_m": z[score_k].tolist(),
            "lego_level_mean_m2_s": lm.tolist(),
            "nemo_level_mean_m2_s": nm.tolist(),
            "lego_over_nemo": ratios.tolist(),
            "original_verdict": classify_offline(nrms, ratios),
            "review_reissued_verdict": classify_offline_review(
                nrms, ratios, z[score_k]),
        }
        if name.startswith("realized_"):
            scored[name]["nemo_evd_fraction_by_level"] = np.mean(
                wm & (nn >= 50.0), axis=0).tolist()

    convention = component_final["kappa_convention"]
    c_lego = component_final["c_k"] * (np.sqrt(2.0) if convention == "gaspar_sqrt2e" else 1.0)
    attribution = closure_attribution(
        A_cl[equator_row, :, 0], avm[equator_row, :, 1],
        component_final["e"][equator_row, :, 0], en_nemo[equator_row, :, 1],
        component_final["l_k"][equator_row, :, 0], mxl_nemo[equator_row, :, 1],
        wet[:, 0], c_lego=c_lego, c_nemo=0.1,
        floor_lego=component_final["floor"], floor_nemo=1.2e-4)
    attribution["depth_m"] = float(z[0])
    attribution["lego_component_call_matches_closure_max_abs"] = component_output_max_abs
    attribution["time_levels"] = component_time_levels

    # Controls proven able to fail: the declared mapping must beat a one-level
    # shift, a wet coefficient planted in a dry cell must be detected, and an
    # empty mask must raise rather than returning a flattering zero.
    base_l, base_n = pairs["closure_avm"]
    ctl_wet, ctl_weights = wet[:, score_k], weights[:, score_k]
    base_nrms = scored["closure_avm"]["nrms"]
    shifted_nrms = weighted_nrms(base_l[:, score_k], wrong_shift(base_n)[:, score_k],
                                 ctl_weights, ctl_wet)
    dry = ~wet
    dry_plant = base_l.copy()
    dry_plant[dry] = 1.0
    dry_violation_count = int(np.count_nonzero(np.abs(dry_plant[dry]) > 0))
    empty_mask_raised = False
    try:
        weighted_nrms(base_l, base_n, weights, np.zeros_like(wet))
    except ValueError:
        empty_mask_raised = True
    controls = {
        "wrong_shift_nrms": shifted_nrms,
        "correct_nrms": base_nrms,
        "wrong_shift_fails": bool(shifted_nrms > base_nrms),
        "dry_cell_plant_detected": bool(dry_violation_count > 0),
        "dry_cell_plant_count": dry_violation_count,
        "empty_mask_raises": empty_mask_raised,
        "evd_nonvacuity_max_abs_Kv_prod_minus_closure": float(np.max(np.abs(K_prod - K_cl))),
        "day0_identity": "verified by _build_twin_state; max dT,deta,du,dv,dbefore,dTKE all 0 in log",
    }
    if not all((controls["wrong_shift_fails"], controls["dry_cell_plant_detected"],
                controls["empty_mask_raises"], controls["evd_nonvacuity_max_abs_Kv_prod_minus_closure"] > 0)):
        raise SystemExit(f"control failure: {controls}")

    input_paths = {
        "restart": Path(dl.restart_path()),
        "mesh_mask": Path(dl.RUN_DIR) / "mesh_mask.nc",
        "nemo_avm_closure": Path(dl.dump_path("tke_dump_avm_final.bin")),
        "nemo_avt_closure": Path(dl.dump_path("tke_dump_avt_final.bin")),
        "nemo_tke_energy": Path(dl.dump_path("tke_dump_en.bin")),
        "nemo_mixing_length": Path(dl.dump_path("tke_dump_zmxlm.bin")),
        "nemo_avm_realized": Path(dl.dump_path("dump_avm.bin")),
        "nemo_avt_realized": Path(dl.dump_path("dump_avt.bin")),
        "namelist": Path(dl.RUN_DIR) / "namelist_cfg",
        "ocean_output": Path(dl.RUN_DIR) / "ocean.output",
        "nemo_binary": Path(dl.RUN_DIR) / "nemo",
        "time_level_registry": ROOT / "packages/ocean/legoesm/ocean/fidelity/time_levels.py",
        "probe_source": Path(__file__).resolve(),
    }
    artifact = {
        "schema": "dino-euc-mechanism-v2-review",
        "headline_verdict": "CONFIRM_VISCOSITY_PRIME_SUSPECT",
        "review_retraction": {
            "old_headline": "vertical mixing not supported at core depth",
            "status": "RETRACTED",
            "reason": (
                "the 10.14 m interface sets the audited 5-26 m shear; its 2.05x "
                "avm excess predicts both weaker shear and a deeper core"
            ),
            "cuda_integrity_accusation": "WITHDRAWN_BY_REVIEWER; original no-CUDA status was honest",
        },
        "state": "shared day-180, first matched step kt=5761",
        "lane": {"name": dl.LANE, "run_dir": dl.RUN_DIR, "restart": dl.RESTART, "kt": dl.KT_DUMP},
        "precision": {"jax_enable_x64": os.environ["JAX_ENABLE_X64"], "state_dtype": str(np.asarray(state.T.data).dtype)},
        "import_receipt": {"module": str(module_path), "within_worktree": True},
        "repo": git_receipt(ROOT),
        "oracle_repo": git_receipt(Path(dl.RUN_DIR).parents[0]),
        "pinned_reducers": pins,
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "source_trace": {
            "nemo_namelist": "RUN_VERDICT360_M0/namelist_cfg:384-401 (TKE+EVD, nn_evdm=1)",
            "nemo_dispatch": "src/OCE/ZDF/zdfphy.F90:264-286,311-323",
            "nemo_tke_mixing_length": "src/OCE/ZDF/zdftke.F90:575-724",
            "nemo_evd": "src/OCE/ZDF/zdfevd.F90:92-120",
            "lego_config": "packages/ocean/legoesm/ocean/experiments/dino.py:2890-2941,3234-3259",
            "lego_call_path": "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6347-6519",
            "lego_closure": "packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py:108",
        },
        "equator": {"row_zero_based": equator_row, "latitude_deg": equator_lat,
                     "depth_window_m": [10.0, 40.0], "weighting": "interface 0.5*(e3t_0[k]+e3t_0[k+1])"},
        "scores": scored,
        "tke_closure_attribution_10m": attribution,
        "controls": controls,
        "short_run": {"status": "DESIGN_ONLY_NO_CUDA_IN_SANDBOX", **substitution_design()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    # Read-back is part of provenance: a malformed/truncated receipt is a hard failure.
    reread = json.loads(out.read_text())
    if reread["input_sha256"] != artifact["input_sha256"]:
        raise SystemExit("provenance read-back mismatch")
    print(json.dumps({"out": str(out), "headline_verdict": artifact["headline_verdict"],
                      "retraction": artifact["review_retraction"],
                      "equator": artifact["equator"],
                      "scores": scored, "controls": controls,
                      "tke_closure_attribution_10m": attribution,
                      "short_run_status": artifact["short_run"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
