#!/usr/bin/env python
"""#1455 seasonal attribution of the circumpolar-channel row-gap rotation.

Pre-registration: ``PREREG_channel_seasonal.md`` at 76622ff2f, committed
before any statistic in this file was computed.  This probe is fully offline.

Usage:
  JAX_ENABLE_X64=1 channel_seasonal.py --self-test
  JAX_ENABLE_X64=1 channel_seasonal.py --out-dir /tmp/dino_channel_seasonal
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR.parent))

HORIZONS = (90, 180, 270, 360)
N_BOOT = 5000
BOOT_SEED = 1226
N_DRIVER_LEGS = 4
CONFIDENCE = 1.0 - 0.05 / N_DRIVER_LEGS

AUDIT_COMMIT = "102ef501a60c915a01f740115124cd99d95f2197"
AUDIT_PATH = "scripts/validate/ocean_fidelity/dino_1226/regional_audit.py"
AUDIT_SHA256 = "2f2bbc03afed792ea0148002029a6ad6481c93bd2ee37679b4bf615aab392fe7"
ATLAS_PATH = "scripts/validate/ocean_fidelity/dino_1226/ts_divergence_atlas.py"
ATLAS_SHA256 = "72c35e61f064c160dae337d3a7e41dfd9ad4d51fd2adbeedb747b258eb231b79"

CHANNEL_COMMIT = "97d3d7188eb5b9fe74a4f505a383342c81af270b"
CHANNEL_PATH = "scripts/validate/ocean_fidelity/dino_1226/channel_rescore.py"
CHANNEL_SHA256 = "a3b51de6292df22b90fa9bd385dcb22ab47a3049c737f2e2abf1a28eb5a18be9"
CHANNEL_ARTIFACT_PATH = "docs/ocean/fidelity/dino_channel_rescore_artifact.json"
CHANNEL_ARTIFACT_SHA256 = "fd39b99842e7c20a5f42b069f8ff732e6c515dee0b0ab00bea1ddae2f52ef11b"

PREREG = "PREREG_channel_seasonal.md@76622ff2f"


def _git_object(commit: str, path: str, expected: str) -> bytes:
    data = subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True,
        capture_output=True,
    ).stdout
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        raise SystemExit(
            f"FATAL: {path} sha256 {got} != pinned {expected}; exact committed "
            "reuse is not proven"
        )
    return data


def _load_git_module(name: str, commit: str, path: str, expected: str):
    data = _git_object(commit, path, expected)
    module = types.ModuleType(name)
    module.__file__ = str(_DIR / Path(path).name)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(data, f"{commit}:{path}", "exec"), module.__dict__)
    return module


# Dependency order is deliberate: regional_audit imports the atlas by name,
# and channel_rescore imports regional_audit by name.  The registered Git
# objects, not a sibling worktree, therefore supply both modules.
X = _load_git_module(
    "ts_divergence_atlas", AUDIT_COMMIT, ATLAS_PATH, ATLAS_SHA256)
R = _load_git_module("regional_audit", AUDIT_COMMIT, AUDIT_PATH, AUDIT_SHA256)
C = _load_git_module(
    "channel_rescore", CHANNEL_COMMIT, CHANNEL_PATH, CHANNEL_SHA256)

N_MEM = C.N_MEM
CHANNEL_ROWS = C.CHANNEL_ROWS
CHANNEL_INDEX = C.CHANNEL_INDEX
PERSIST_HI = C.PERSIST_HI
PERSIST_LO = C.PERSIST_LO


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def _git_state():
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    return sha, dirty


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_close(a, b, tol=1e-12):
    if not abs(float(a) - float(b)) <= tol:
        raise AssertionError(f"{a} != {b} within {tol}")


def _plant_fires(label, fn):
    try:
        fn()
    except (AssertionError, SystemExit, ValueError):
        print(f"  PLANT FIRED: {label}")
        return
    raise AssertionError(f"planted violation did not fire: {label}")


def _wind_signature_guard(fn):
    names = {name.lower() for name in inspect.signature(fn).parameters}
    if any("time" in name or "seconds" in name or name in {"t", "kt"}
           for name in names):
        raise ValueError(f"wind callable is time-dependent by signature: {names}")
    return sorted(names)


def wind_receipts():
    from legoesm.ocean.experiments import dino

    signature = str(inspect.signature(dino.dino_wind_stress))
    parameters = _wind_signature_guard(dino.dino_wind_stress)
    card_source = Path(dino.__file__)
    source = inspect.getsource(dino.apply_dino_lat_lon_surface_forcing)
    if "dino_T_star_seasonal" not in source or "dino_Q_sr_seasonal" not in source:
        raise SystemExit("FATAL: live DINO annual-cycle path no longer names T* and Qsr")
    if "dino_wind_stress" in source:
        raise SystemExit("FATAL: live DINO time-dependent path now recomputes wind")

    namelist = Path(R.X.nemo_dir(0)) / "namelist_cfg"
    oracle_source = Path(R.A.DINO) / "MY_SRC" / "usrdef_sbc.F90"
    nl_text = namelist.read_text()
    src_text = oracle_source.read_text()
    required = (
        "nn_forcingtype       =      4",
        "ln_ann_cyc        = .true.",
    )
    if not all(token in nl_text for token in required):
        raise SystemExit(
            "FATAL: oracle namelist no longer selects DINO forcing type 4 "
            "annual cycle")
    utau_lines = [line.strip() for line in src_text.splitlines()
                  if "utau(ji,jj)" in line and "=" in line]
    if not any("znl_cbc(znds_wnd_phi, znds_wnd_val, gphiu" in line
               for line in utau_lines):
        raise SystemExit("FATAL: active oracle wind assignment was not found")
    if any("zcos_sais" in line for line in utau_lines):
        raise SystemExit("FATAL: oracle wind assignment now contains seasonal phase")
    return {
        "status": "EXCLUDED_A_PRIORI_TIME_INDEPENDENT",
        "remaining_drivers": [
            "density_inventory", "vertical_stratification",
            "mixed_layer_depth", "thermal_wind",
        ],
        "legoesm_wind_signature": signature,
        "legoesm_wind_parameters": parameters,
        "legoesm_card_path": str(card_source),
        "legoesm_card_sha256": _sha_file(card_source),
        "oracle_namelist": str(namelist),
        "oracle_namelist_sha256": _sha_file(namelist),
        "oracle_usrdef_sbc": str(oracle_source),
        "oracle_usrdef_sbc_sha256": _sha_file(oracle_source),
        "oracle_utau_assignments": utau_lines,
        "reason": (
            "the card's annual-cycle branch recomputes T* and Qsr only; both "
            "model wind functions depend on latitude and fixed knots, not time"
        ),
    }


def _weighted_phase_stats(target, driver, shift):
    q = np.roll(_f64(driver), -shift, axis=0)
    x = _f64(target)
    correlations = []
    n_eff = []
    denominators = []
    for it in range(len(HORIZONS)):
        r, den = C.pearson(x[it], q[it])
        ne, ne_den = C.effective_rows(x[it], q[it])
        correlations.append(r)
        n_eff.append(ne)
        denominators.append({"pearson": den, "bartlett": ne_den})
    correlations = np.asarray(correlations, dtype=np.float64)
    n_eff = np.asarray(n_eff, dtype=np.float64)
    weights = np.maximum(n_eff - 3.0, 0.0)
    if not float(weights.sum()) > 0.0:
        raise ValueError("phase score UNMEASURABLE: no effective rows above 3")
    z = np.arctanh(np.clip(correlations, -1.0 + 1e-12, 1.0 - 1e-12))
    zbar = float(np.sum(weights * z) / np.sum(weights))
    return {
        "score": abs(float(np.tanh(zbar))),
        "signed_r": float(np.tanh(zbar)),
        "correlations": correlations.tolist(),
        "n_eff": n_eff.tolist(),
        "n_eff_median": float(np.median(n_eff)),
        "n_eff_min": float(np.min(n_eff)),
        "weights": weights.tolist(),
        "denominators": denominators,
    }


def phase_decider(target, driver, *, n_boot=N_BOOT, seed=BOOT_SEED,
                  confidence=CONFIDENCE):
    """Correct phase versus every non-zero cyclic shift, with spatial blocks."""
    x = _f64(target)
    q = _f64(driver)
    if x.shape != q.shape or x.ndim != 2 or x.shape[0] != len(HORIZONS):
        raise ValueError(f"phase profiles require shape (4,n), got {x.shape}/{q.shape}")
    by_shift = {str(s): _weighted_phase_stats(x, q, s) for s in range(4)}
    matched = by_shift["0"]
    best_shift = max(range(1, 4), key=lambda s: by_shift[str(s)]["score"])
    best_null = by_shift[str(best_shift)]["score"]
    orientation = 1.0 if matched["signed_r"] >= 0.0 else -1.0
    oriented = orientation * np.asarray(matched["correlations"])
    delta = matched["score"] - best_null

    n = x.shape[1]
    block = int(np.clip(np.ceil(n / matched["n_eff_median"]), 2, 12))
    rng = np.random.default_rng(seed)
    draws = []
    dropped = 0
    for _ in range(n_boot):
        idx = C.moving_block_indices(n, block, rng)
        try:
            scores = [_weighted_phase_stats(x[:, idx], q[:, idx], s)["score"]
                      for s in range(4)]
            draws.append(scores[0] - max(scores[1:]))
        except ValueError:
            dropped += 1
    if len(draws) < n_boot // 2:
        raise RuntimeError("phase bootstrap retained fewer than half its draws")
    alpha = 1.0 - confidence
    ci = np.percentile(draws, [50.0 * alpha, 100.0 - 50.0 * alpha])
    if (matched["score"] >= PERSIST_HI
            and np.all(oriented > PERSIST_LO) and ci[0] > 0.0):
        status = "CONFIRMS_PHASE_TRACKING"
    elif matched["score"] < PERSIST_LO or ci[1] < 0.0:
        status = "REFUTES_PHASE_TRACKING"
    else:
        status = "UNRESOLVED"
    return {
        "status": status,
        "correct_phase_score": matched["score"],
        "correct_phase_signed_r": matched["signed_r"],
        "correct_phase_oriented_horizon_r": oriented.tolist(),
        "cyclic_scores": {key: value["score"] for key, value in by_shift.items()},
        "best_null_shift_quarters": best_shift,
        "best_null_score": best_null,
        "delta_vs_best_cyclic": delta,
        "delta_block_ci": [float(ci[0]), float(ci[1])],
        "confidence": confidence,
        "block_length_rows": block,
        "bootstrap_requested": n_boot,
        "bootstrap_retained": len(draws),
        "bootstrap_dropped": dropped,
        "n_positions": n,
        "n_eff_median": matched["n_eff_median"],
        "n_eff_min": matched["n_eff_min"],
        "by_shift": by_shift,
    }


def phase_collection(target, driver, *, n_boot=N_BOOT, seed=BOOT_SEED):
    """Score four paired members and their ensemble-mean profiles."""
    x, q = _f64(target), _f64(driver)
    if x.shape != q.shape or x.shape[:2] != (N_MEM, len(HORIZONS)):
        raise ValueError(f"phase collection requires (4,4,n), got {x.shape}/{q.shape}")
    mean = phase_decider(x.mean(axis=0), q.mean(axis=0), n_boot=n_boot,
                         seed=seed)
    members = [phase_decider(x[m], q[m], n_boot=n_boot,
                             seed=seed + 10 + m) for m in range(N_MEM)]
    member_status = [entry["status"] for entry in members]
    if (mean["status"] == "CONFIRMS_PHASE_TRACKING"
            and all(s == "CONFIRMS_PHASE_TRACKING" for s in member_status)):
        status = "CONFIRMS_PHASE_TRACKING"
    elif (mean["status"] == "REFUTES_PHASE_TRACKING"
          and sum(s == "REFUTES_PHASE_TRACKING" for s in member_status) >= 3):
        status = "REFUTES_PHASE_TRACKING"
    else:
        status = "UNRESOLVED"
    return {
        "status": status, "ensemble_mean": mean, "members": members,
        "member_statuses": member_status,
        "confirm_rule": "ensemble mean and all 4 members",
        "refute_rule": "ensemble mean and at least 3 of 4 members",
    }


def seasonal_anomaly(a):
    x = _f64(a)
    return x - np.mean(x, axis=1, keepdims=True)


def proportionality(target, predictor):
    d, c = _f64(target), _f64(predictor)
    den = float(np.sum(c * c))
    target_energy = float(np.sum(d * d))
    if not (den > 0.0 and target_energy > 0.0):
        raise ValueError("fixed-fraction fit UNMEASURABLE on zero energy")
    beta = float(np.sum(c * d) / den)
    nrmse = float(np.sqrt(np.sum((d - beta * c) ** 2) / target_energy))
    return {"beta": beta, "nrmse": nrmse,
            "predictor_energy": den, "target_energy": target_energy}


def trivial_explanation(transport_lego, transport_nemo, *, n_boot=N_BOOT):
    lego = seasonal_anomaly(transport_lego)
    nemo = seasonal_anomaly(transport_nemo)
    gap = lego - nemo
    common = 0.5 * (lego + nemo)
    same = phase_collection(lego, nemo, n_boot=n_boot, seed=BOOT_SEED + 100)
    fraction = phase_collection(gap, common, n_boot=n_boot, seed=BOOT_SEED + 200)
    pooled = proportionality(gap.mean(axis=0), common.mean(axis=0))
    members = [proportionality(gap[m], common[m]) for m in range(N_MEM)]
    all_small = pooled["nrmse"] <= PERSIST_LO and all(
        item["nrmse"] <= PERSIST_LO for item in members)
    mostly_large = (pooled["nrmse"] >= PERSIST_HI
                    and sum(item["nrmse"] >= PERSIST_HI for item in members) >= 3)
    if (same["status"] == "CONFIRMS_PHASE_TRACKING"
            and fraction["status"] == "CONFIRMS_PHASE_TRACKING" and all_small):
        status = "CONFIRMS_FIXED_FRACTION_OF_SHARED_ROTATING_FIELD"
    elif (same["status"] == "REFUTES_PHASE_TRACKING"
          or fraction["status"] == "REFUTES_PHASE_TRACKING" or mostly_large):
        status = "REFUTES_FIXED_FRACTION_EXPLANATION"
    else:
        status = "UNRESOLVED"
    return {
        "status": status, "same_way_rotation": same,
        "gap_tracks_common_field": fraction,
        "pooled_fixed_fraction": pooled, "member_fixed_fraction": members,
        "nrmse_confirm_bar": PERSIST_LO, "nrmse_refute_bar": PERSIST_HI,
    }


def density_profiles(rho, wet):
    """Volume-weighted row density inventory and vertical regression slope."""
    volume = _f64(X.CELL_VOL)
    w = np.where(wet, volume, 0.0)[CHANNEL_ROWS]
    r = np.where(wet, _f64(rho), 0.0)[CHANNEL_ROWS]
    z = _f64(R.A.gdept0)[CHANNEL_ROWS]
    total = np.sum(w, axis=(1, 2))
    if np.any(total <= 0.0):
        raise ValueError("density driver has an empty channel row")
    rbar = np.sum(w * r, axis=(1, 2)) / total
    zbar = np.sum(w * z, axis=(1, 2)) / total
    dr = r - rbar[:, None, None]
    dz = z - zbar[:, None, None]
    slope_den = np.sum(w * dz * dz, axis=(1, 2))
    if np.any(slope_den <= 0.0):
        raise ValueError("stratification driver has zero weighted depth variance")
    slope = np.sum(w * dz * dr, axis=(1, 2)) / slope_den
    return _f64(rbar), _f64(slope)


def _seos_surface(temperature, salinity, pressure):
    return R.A.nemo_seos_eos(temperature, salinity, pressure, R.A.CFG)


def mld_profile(st, wet):
    from legoesm.ocean.diagnostics import mixed_layer_depth

    thickness = np.where(wet, _f64(R.A.e3t0), 0.0)
    bottom = np.sum(thickness, axis=-1)
    mld = _f64(mixed_layer_depth(
        _f64(st["T"]), _f64(st["S"]), _f64(R.A.gdept1d),
        delta_sigma=0.01, ref_depth_m=10.0, wet_mask=wet,
        bottom_depth=bottom, eos_fn=_seos_surface, p_ref_pa=0.0,
    ))
    area = _f64(X.e1t) * _f64(X.e2t)
    surface_w = np.where(wet[:, :, 0], area, 0.0)[CHANNEL_ROWS]
    values = mld[CHANNEL_ROWS]
    den = np.sum(surface_w, axis=1)
    if np.any(den <= 0.0):
        raise ValueError("MLD driver has an empty channel row")
    return _f64(np.sum(surface_w * values, axis=1) / den)


def thermal_wind_profile(rho, wet):
    values = []
    for j in range(CHANNEL_ROWS.start, CHANNEL_ROWS.stop - 1):
        per_lon, _ = R.A.thermal_wind_rows(rho, wet, j, j + 1)
        values.append(float(R.D._avg(per_lon)))
    return _f64(values)


def _finite_fp64(name, *arrays):
    for a in arrays:
        x = np.asarray(a)
        if x.dtype != np.float64:
            raise SystemExit(f"FATAL: {name} dtype {x.dtype}, expected fp64")
        if not np.all(np.isfinite(x)):
            raise SystemExit(f"FATAL: {name} contains non-finite values")


def upstream_artifact():
    data = _git_object(
        CHANNEL_COMMIT, CHANNEL_ARTIFACT_PATH, CHANNEL_ARTIFACT_SHA256)
    return json.loads(data)


def verify_upstream_gaps(transport_lego, transport_nemo, artifact):
    gaps = transport_lego - transport_nemo
    expected = np.asarray([
        artifact["gaps_paired_sv"][str(day)] for day in HORIZONS
    ], dtype=np.float64).transpose(1, 0, 2)
    np.testing.assert_allclose(gaps, expected, rtol=0.0, atol=5e-13)
    return float(np.max(np.abs(gaps - expected)))


def saturation_summary(lego, nemo):
    """Exact regional-audit saturation engine, position by position."""
    rows = []
    for j in range(lego.shape[-1]):
        history = {
            (side, day): float(np.std(values[:, it, j], ddof=1))
            for side, values in (("lego", lego), ("nemo", nemo))
            for it, day in enumerate(HORIZONS)
        }
        saturated, reason = R.saturated(history)
        rows.append({
            "position": j, "saturated": bool(saturated), "reason": reason,
            "spreads": {f"{side}_day{day}": history[(side, day)]
                        for side in ("lego", "nemo") for day in HORIZONS},
        })
    n_sat = sum(item["saturated"] for item in rows)
    return {"saturated": n_sat, "unsaturated": len(rows) - n_sat,
            "denominator": len(rows), "rows": rows}


def _collect_state_drivers(st, wet):
    rho = _f64(R.A.rho_of(st, wet))
    density, stratification = density_profiles(rho, wet)
    mld = mld_profile(st, wet)
    thermal = thermal_wind_profile(rho, wet)
    _finite_fp64("state drivers", density, stratification, mld, thermal)
    return {
        "density_inventory": density,
        "vertical_stratification": stratification,
        "mixed_layer_depth": mld,
        "thermal_wind": thermal,
    }


def load_measurements():
    clock_compat = C._install_clock_helper_compat()
    provenance_receipt, clock_return = R.control_stamps()
    restart = f"{R.X.nemo_dir(0)}/DINO_{R.X.G.KT_RESTART:08d}_restart.nc"
    clock_seconds = float(R.X.T.restart_elapsed_seconds(restart))
    if clock_seconds != 15552000.0:
        raise SystemExit(f"FATAL: oracle clock is {clock_seconds}, expected 15552000")

    transport = {
        "lego": np.empty((N_MEM, len(HORIZONS), C.N_ROWS), dtype=np.float64),
        "nemo": np.empty((N_MEM, len(HORIZONS), C.N_ROWS), dtype=np.float64),
    }
    shapes = {"density_inventory": C.N_ROWS,
              "vertical_stratification": C.N_ROWS,
              "mixed_layer_depth": C.N_ROWS,
              "thermal_wind": C.N_ROWS - 1}
    drivers = {
        name: {side: np.empty((N_MEM, len(HORIZONS), n), dtype=np.float64)
               for side in ("lego", "nemo")}
        for name, n in shapes.items()
    }
    land = None
    for it, day in enumerate(HORIZONS):
        for member in range(N_MEM):
            print(f"loading drivers day {day} member {member}", flush=True)
            lego = R.load_lego(member, day)
            nemo = R.load_nemo(member, day)
            if land is None:
                land = _f64(lego["land_mask"])
            elif not np.array_equal(land, _f64(lego["land_mask"])):
                raise SystemExit("FATAL: legoESM land mask changed across states")
            wet = R.A.tmask & (land[:, :, None] > 0.5)
            for side, st in (("lego", lego), ("nemo", nemo)):
                transport[side][member, it] = R.row_transports(
                    st["u"], R.A.umask)[CHANNEL_ROWS]
                reduced = _collect_state_drivers(st, wet)
                for name, value in reduced.items():
                    drivers[name][side][member, it] = value
    _finite_fp64("transport", transport["lego"], transport["nemo"])
    return transport, drivers, {
        "upstream_control_provenance_return": provenance_receipt,
        "upstream_control_clock_return_was_null": clock_return is None,
        "clock_helper_compat": clock_compat,
        "oracle_restart": restart,
        "oracle_elapsed_seconds": clock_seconds,
        "oracle_elapsed_days": clock_seconds / 86400.0,
        "lego_launch_sha": Path(f"{R.X.LEGO_DIR}/.launch_sha").read_text().strip(),
        "lego_members": [R.X.lego_npz(i) for i in range(N_MEM)],
        "nemo_member_directories": [os.path.realpath(R.X.nemo_dir(i))
                                    for i in range(N_MEM)],
    }


def score_driver(name, values, target_gap, *, n_boot=N_BOOT):
    driver_gap = values["lego"] - values["nemo"]
    driver_anomaly = seasonal_anomaly(driver_gap)
    target = target_gap
    if name == "thermal_wind":
        target = 0.5 * (target[:, :, :-1] + target[:, :, 1:])
    phase = phase_collection(target, driver_anomaly, n_boot=n_boot,
                             seed=BOOT_SEED + 1000 * (1 + list(
                                 ("density_inventory", "vertical_stratification",
                                  "mixed_layer_depth", "thermal_wind")).index(name)))
    saturation = saturation_summary(values["lego"], values["nemo"])
    suffix = " (u)" if saturation["unsaturated"] else ""
    return {
        "status": phase["status"] + suffix,
        "phase": phase,
        "saturation": saturation,
        "driver_gap": driver_gap.tolist(),
        "driver_seasonal_anomaly": driver_anomaly.tolist(),
        "weighting": {
            "density_inventory": "e1t*e2t*e3t_0 volume",
            "vertical_stratification": "e1t*e2t*e3t_0 weighted regression",
            "mixed_layer_depth": "e1t*e2t surface area after canonical 0.01/10m MLD",
            "thermal_wind": "thermal_wind_rows e3t_0*gdept_0, then recorded longitude mean",
        }[name],
    }


def self_test(n_boot=300):
    print("CHANNEL SEASONAL SELF-TEST")
    _plant_fires(
        "pinned source hash mutation",
        lambda: _git_object(AUDIT_COMMIT, AUDIT_PATH, "0" * 64),
    )
    from legoesm.ocean.experiments.dino import dino_wind_stress
    _wind_signature_guard(dino_wind_stress)
    _plant_fires(
        "time-dependent wind signature",
        lambda: _wind_signature_guard(
            lambda lat_deg, t_seconds, cfg=None: lat_deg + t_seconds),
    )

    wet = R.A.tmask.copy()
    rho = np.where(wet, 1027.0, np.nan)
    rho[:, :, 0] += 0.1
    official, _ = density_profiles(rho, wet)
    layer = np.mean(np.where(wet[CHANNEL_ROWS], rho[CHANNEL_ROWS], 0.0),
                    axis=(1, 2))
    _plant_fires(
        "layer average cannot replace volume-weighted density",
        lambda: np.testing.assert_allclose(layer, official, rtol=1e-12, atol=1e-12),
    )

    from legoesm.ocean.diagnostics import mixed_layer_depth
    z = np.asarray([5.0, 15.0, 30.0, 60.0], dtype=np.float64)
    t_cross = np.asarray([[0.0, 0.0, 0.02, 0.04]], dtype=np.float64)
    t_flat = np.zeros_like(t_cross)
    salinity = np.zeros_like(t_cross)

    def synthetic_eos(temperature, salinity, pressure):
        del salinity, pressure
        return 1000.0 + temperature

    cross = float(np.asarray(mixed_layer_depth(
        t_cross, salinity, z, delta_sigma=0.01,
        wet_mask=np.ones_like(t_cross), bottom_depth=np.asarray([80.0]),
        eos_fn=synthetic_eos))[0])
    flat = float(np.asarray(mixed_layer_depth(
        t_flat, salinity, z, delta_sigma=0.01,
        wet_mask=np.ones_like(t_flat), bottom_depth=np.asarray([80.0]),
        eos_fn=synthetic_eos))[0])
    if not abs(cross - flat) > 1.0:
        raise AssertionError("MLD plant did not move the canonical diagnostic")
    _plant_fires("MLD crossing is mobile", lambda: _assert_close(cross, flat, 1.0))

    constant = np.where(wet, 1027.0, np.nan)
    base = thermal_wind_profile(constant, wet)
    moved_rho = constant.copy()
    moved_rho[CHANNEL_ROWS.start + 1] += 0.01
    moved = thermal_wind_profile(moved_rho, wet)
    if not np.max(np.abs(moved - base)) > 0.0:
        raise AssertionError("thermal-wind plant did not move")
    _plant_fires("meridional density perturbation moves thermal wind",
                 lambda: np.testing.assert_allclose(moved, base, atol=0.0))

    block_profile = np.repeat(np.arange(7, dtype=np.float64), 5)
    n_eff, _ = C.effective_rows(block_profile, block_profile)
    _plant_fires("adjacent rows are not 35 independent rows",
                 lambda: _assert_close(n_eff, 35.0))

    rng = np.random.default_rng(1455)
    patterns = rng.normal(size=(len(HORIZONS), C.N_ROWS))
    known = phase_decider(patterns, patterns, n_boot=n_boot, seed=77)
    shifted = phase_decider(patterns, np.roll(patterns, 1, axis=0),
                            n_boot=n_boot, seed=78)
    if known["status"] != "CONFIRMS_PHASE_TRACKING":
        raise AssertionError(f"known aligned phases classified {known['status']}")
    _plant_fires(
        "one-quarter relabelling cannot keep the aligned verdict",
        lambda: (_ for _ in ()).throw(AssertionError())
        if shifted["status"] != known["status"] else None,
    )
    print("CHANNEL SEASONAL SELF-TEST PASSED -- 7/7 local plants fired")


def run(out_dir):
    sha_start, dirty_start = _git_state()
    if dirty_start:
        raise SystemExit(f"FATAL: producer tree {sha_start} is dirty")
    self_test()
    wind = wind_receipts()
    upstream = upstream_artifact()
    transport, drivers, input_provenance = load_measurements()
    upstream_residual = verify_upstream_gaps(
        transport["lego"], transport["nemo"], upstream)
    planted = transport["lego"].copy()
    planted[0, 0, 0] += 1e-6
    _plant_fires(
        "upstream row-gap reproduction",
        lambda: verify_upstream_gaps(planted, transport["nemo"], upstream),
    )

    transport_gap_anomaly = seasonal_anomaly(
        transport["lego"] - transport["nemo"])
    print("scoring trivial fixed-fraction explanation first", flush=True)
    trivial = trivial_explanation(
        transport["lego"], transport["nemo"], n_boot=N_BOOT)
    scored = {}
    for name in ("density_inventory", "vertical_stratification",
                 "mixed_layer_depth", "thermal_wind"):
        print(f"scoring driver {name}", flush=True)
        scored[name] = score_driver(
            name, drivers[name], transport_gap_anomaly, n_boot=N_BOOT)

    sha_end, dirty_end = _git_state()
    if (sha_end, dirty_end) != (sha_start, dirty_start):
        raise SystemExit("FATAL: producer tree changed during measurement")
    result = {
        "trivial_explanation_first": trivial,
        "wind": wind,
        "drivers": scored,
        "transport": {
            "lego_row_transport_sv": transport["lego"].tolist(),
            "nemo_row_transport_sv": transport["nemo"].tolist(),
            "gap_seasonal_anomaly_sv": transport_gap_anomaly.tolist(),
            "upstream_gap_max_abs_reproduction_residual_sv": upstream_residual,
            "target_saturation": upstream["_stamp"]["floor_saturation_measured"],
            "target_label": (
                "CONFIRMED coherent seasonal reorganization; amplitude remains "
                "PLAUSIBLE because 0/35 row floors are saturated"
            ),
        },
        "_stamp": {
            "producer_sha": sha_start,
            "producer_tree_dirty": dirty_start,
            "probe": Path(__file__).name,
            "prereg": PREREG,
            "horizons": list(HORIZONS),
            "members_per_side": N_MEM,
            "channel_rows": [int(CHANNEL_INDEX.min()), int(CHANNEL_INDEX.max())],
            "pinned_imports": {
                "regional_audit": {"commit": AUDIT_COMMIT, "sha256": AUDIT_SHA256},
                "ts_divergence_atlas": {"commit": AUDIT_COMMIT, "sha256": ATLAS_SHA256},
                "channel_rescore": {"commit": CHANNEL_COMMIT, "sha256": CHANNEL_SHA256},
                "channel_artifact": {"commit": CHANNEL_COMMIT,
                                     "sha256": CHANNEL_ARTIFACT_SHA256},
            },
            "input_provenance": input_provenance,
            "bars": {
                "PERSIST_HI": PERSIST_HI, "PERSIST_LO": PERSIST_LO,
                "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED,
                "N_DRIVER_LEGS": N_DRIVER_LEGS,
                "familywise_confidence": CONFIDENCE,
            },
            "weighting": "thickness/volume weighted throughout; no layer average",
            "wind_excluded_before_driver_scoring": True,
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "channel_seasonal.json"
    path.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({
        "trivial": trivial["status"],
        "wind": wind["status"],
        "drivers": {name: value["status"] for name, value in scored.items()},
        "artifact": str(path), "producer_sha": sha_start,
    }, indent=1))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/dino_channel_seasonal")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
    else:
        run(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
