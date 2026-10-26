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
    _nemo_bn2_zrw,
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


def nemo_literal_tke_recurrence(
    a_ext: np.ndarray, b_ext: np.ndarray, c_ext: np.ndarray,
    rhs_ext: np.ndarray, surface_en: np.ndarray,
    w_active: np.ndarray, floor: float,
) -> np.ndarray:
    """Post-hoc row-17 discriminator for zdftke.F90:547-565 source order.

    This is deliberately a NumPy scorer, not production physics.  It preserves
    NEMO's reciprocal surface seed ``zdiag(1)=1/en(1), zd_lw(1)=1``, stops the
    forward recurrence at ``jpkm1``, seeds that row directly, back-substitutes
    through level 2, and only then applies the floor/mask.
    """
    diag, work = nemo_literal_tke_forward_stages(
        a_ext, b_ext, c_ext, rhs_ext, surface_en)
    up = np.asarray(c_ext)
    rhs = np.asarray(rhs_ext)
    n_ext = diag.shape[-1]
    last_solved = n_ext - 2
    solved = np.array(rhs, copy=True)
    solved[..., last_solved] = (
        work[..., last_solved] / diag[..., last_solved])
    for k in range(last_solved - 1, 0, -1):
        solved[..., k] = (
            work[..., k] - up[..., k] * solved[..., k + 1]
        ) / diag[..., k]
    interior = solved[..., 1:]
    return np.maximum(interior, np.float64(floor)) * np.asarray(w_active)


def nemo_literal_tke_forward_stages(
    a_ext: np.ndarray, b_ext: np.ndarray, c_ext: np.ndarray,
    rhs_ext: np.ndarray, surface_en: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal zdftke.F90:560-569 forward diagonal and RHS recurrences."""
    a = np.asarray(a_ext)
    diag = np.array(b_ext, copy=True)
    up = np.asarray(c_ext)
    rhs = np.asarray(rhs_ext)
    work = np.array(rhs, copy=True)
    last_solved = diag.shape[-1] - 2
    diag[..., 0] = 1.0 / np.asarray(surface_en)
    work[..., 0] = 1.0
    for k in range(1, last_solved + 1):
        diag[..., k] = (
            diag[..., k]
            - a[..., k] * up[..., k - 1] / diag[..., k - 1])
    for k in range(1, last_solved + 1):
        work[..., k] = (
            rhs[..., k]
            - a[..., k] / diag[..., k - 1] * work[..., k - 1])
    return diag, work


def exact_difference_census(
    lego: np.ndarray,
    nemo: np.ndarray,
    wet: np.ndarray,
    focus: list[tuple[int, int]],
) -> dict:
    """Bitwise companion to the registered tolerance bar.

    This does not change a row's VERIFIED/DIVERGED disposition.  It names the
    first numeric operand difference that a later division may amplify.
    """
    lego = np.asarray(lego)
    nemo = np.asarray(nemo)
    wet = np.asarray(wet, dtype=bool)
    unequal = wet & ~((lego == nemo) | (np.isnan(lego) & np.isnan(nemo)))
    bad_columns = np.any(unequal, axis=-1)
    abs_diff = np.where(wet, np.abs(lego - nemo), 0.0)
    return {
        "n_unequal_wet_elements": int(unequal.sum()),
        "n_unequal_wet_columns": int(bad_columns.sum()),
        "max_absolute_difference": float(np.max(abs_diff)),
        "focus": [
            {"j": j, "i": i, "exact": bool(not bad_columns[j, i])}
            for j, i in focus
        ],
        "bit_identical": bool(not unequal.any()),
    }


def restart_numeric_identity(instrument: Path, other: Path) -> dict:
    """Compare NetCDF numeric state, deliberately ignoring container bytes."""
    with xr.open_dataset(instrument, decode_times=False) as left, \
            xr.open_dataset(other, decode_times=False) as right:
        left_numeric = {
            name for name in left.variables
            if np.issubdtype(left[name].dtype, np.number)
        }
        right_numeric = {
            name for name in right.variables
            if np.issubdtype(right[name].dtype, np.number)
        }
        common = sorted(left_numeric & right_numeric)
        differences = []
        for name in common:
            a = np.asarray(left[name])
            b = np.asarray(right[name])
            same = (a.shape == b.shape
                    and np.array_equal(a, b, equal_nan=True))
            if not same:
                differences.append({
                    "name": name,
                    "left_shape": list(a.shape),
                    "right_shape": list(b.shape),
                    "max_absolute_difference": (
                        float(np.nanmax(np.abs(a - b)))
                        if a.shape == b.shape else None),
                })
        return {
            "instrument": str(instrument),
            "other": str(other),
            "instrument_numeric_variables": len(left_numeric),
            "other_numeric_variables": len(right_numeric),
            "missing_from_other": sorted(left_numeric - right_numeric),
            "missing_from_instrument": sorted(right_numeric - left_numeric),
            "common_numeric_variables": len(common),
            "n_different_common_variables": len(differences),
            "differences": differences,
            "common_variables_bit_identical": not differences,
            "same_numeric_variable_set": left_numeric == right_numeric,
            "pass": not differences and left_numeric == right_numeric,
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
    real_tri = tke_mod._tridiag_thomas
    real_literal_tri = tke_mod._nemo_literal_tke_solve
    real_lc = tke_mod.nemo_langmuir_tke_source
    real_etau = tke_mod.nemo_etau_injection
    real_prandtl = tke_mod._prandtl_number
    calls = []
    mxl_calls = []
    solve_calls = []
    tri_calls = []
    literal_tri_calls = []
    lc_calls = []
    etau_calls = []
    prandtl_calls = []

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

    def spy_tri(a, b, c, d):
        out = real_tri(a, b, c, d)
        tri_calls.append(((a, b, c, d), out))
        return out

    def spy_literal_tri(*args, **kwargs):
        out = real_literal_tri(*args, **kwargs)
        literal_tri_calls.append((args, kwargs, out))
        return out

    def spy_lc(*args, **kwargs):
        out = real_lc(*args, **kwargs)
        lc_calls.append((args, kwargs, out))
        return out

    def spy_etau(*args, **kwargs):
        out = real_etau(*args, **kwargs)
        etau_calls.append((args, kwargs, out))
        return out

    def spy_prandtl(*args, **kwargs):
        out = real_prandtl(*args, **kwargs)
        prandtl_calls.append((args, kwargs, out))
        return out

    shared.avm_weighted_shear_production = spy
    tke_mod.compute_mixing_lengths = spy_mxl
    tke_mod._solve_tke_backward_euler = spy_solve
    tke_mod._tridiag_thomas = spy_tri
    tke_mod._nemo_literal_tke_solve = spy_literal_tri
    tke_mod.nemo_langmuir_tke_source = spy_lc
    tke_mod.nemo_etau_injection = spy_etau
    tke_mod._prandtl_number = spy_prandtl
    try:
        with jax.disable_jit():
            model.step(state, kamm.DT, surface_forcing=forcing)
    finally:
        shared.avm_weighted_shear_production = real
        tke_mod.compute_mixing_lengths = real_mxl
        tke_mod._solve_tke_backward_euler = real_solve
        tke_mod._tridiag_thomas = real_tri
        tke_mod._nemo_literal_tke_solve = real_literal_tri
        tke_mod.nemo_langmuir_tke_source = real_lc
        tke_mod.nemo_etau_injection = real_etau
        tke_mod._prandtl_number = real_prandtl
    if not calls:
        raise AssertionError("avm_weighted_shear_production never fired")
    if not mxl_calls or not solve_calls or not lc_calls or not prandtl_calls:
        raise AssertionError("TKE mixing-length/Langmuir/solve stages never fired")
    return real, calls[0], {
        "mxl_calls": mxl_calls, "lc_calls": lc_calls,
        "etau_calls": etau_calls,
        "solve_calls": solve_calls, "tri_calls": tri_calls,
        "literal_tri_calls": literal_tri_calls,
        "prandtl_calls": prandtl_calls,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mld-maps", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bn2-chain-dir", type=Path)
    ap.add_argument("--bn2-bracket-off-restart", type=Path)
    ap.add_argument("--bn2-bracket-instrument-restart", type=Path)
    ap.add_argument("--bn2-certified-restart", type=Path)
    ap.add_argument("--row17-postsolve-dump", type=Path)
    ap.add_argument("--row17-forward-diag-dump", type=Path)
    ap.add_argument("--row17-forward-rhs-dump", type=Path)
    ap.add_argument("--row17-pre-diag-dump", type=Path)
    ap.add_argument("--row17-pre-lower-dump", type=Path)
    ap.add_argument("--row17-pre-rhs-dump", type=Path)
    ap.add_argument("--row17-postlc-dump", type=Path)
    ap.add_argument("--row17-bracket-off-restart", type=Path)
    ap.add_argument("--row17-bracket-instrument-restart", type=Path)
    ap.add_argument("--row17-instrument-source", type=Path)
    ap.add_argument("--row17-instrument-binary", type=Path)
    ap.add_argument("--row18-argument-dump", type=Path)
    ap.add_argument("--row18-exp-dump", type=Path)
    ap.add_argument("--row18-increment-dump", type=Path)
    ap.add_argument("--row18-bracket-instrument-restart", type=Path)
    ap.add_argument("--row18-instrument-source", type=Path)
    ap.add_argument("--row18-instrument-binary", type=Path)
    args = ap.parse_args()
    bracket_args = (
        args.bn2_bracket_off_restart,
        args.bn2_bracket_instrument_restart,
        args.bn2_certified_restart,
    )
    if args.bn2_chain_dir is not None and any(x is None for x in bracket_args):
        raise SystemExit(
            "--bn2-chain-dir requires both bracket restarts and the certified restart")

    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit(f"CPU/fp64 required; backend={jax.default_backend()} x64={jax.config.x64_enabled}")
    focus = focus_from_maps(args.mld_maps)
    st = base.build_state()
    jpi, jpj, jpk, hls = st["jpi"], st["jpj"], st["jpk"], st["hls"]
    nj, ni = st["ni_ni"]
    T, S = st["T"], st["S"]
    gdept, gdepw, e3w = nemo_bn2_live_geometry(
        st["z_coord"], st["eta"], st["H_bathy"],
        r3t_evaluation="nemo_reciprocal")
    zrw_gdept_0 = getattr(st["z_coord"], "nemo_gdept_0", None)
    zrw_gdepw_0 = getattr(st["z_coord"], "nemo_gdepw_0", None)
    if zrw_gdept_0 is None or zrw_gdepw_0 is None:
        raise AssertionError(
            "DINO row-11 literal zrw requires raw gdept_0/gdepw_0")
    # The stored raw W ladder includes NEMO's zero-depth surface point, while
    # eosbn2's jk=2 first evaluated interface is array index one.
    zrw_gdepw_0 = zrw_gdepw_0[..., 1:]
    zrw_stretch = nemo_r3t_stretch(
        st["z_coord"], st["eta"], st["H_bathy"],
        evaluation="nemo_reciprocal")
    gdept = jnp.asarray(gdept, dtype=T.dtype)
    gdepw = jnp.asarray(gdepw, dtype=T.dtype)
    alpha, beta = nemo_seos_alpha_beta(T, S, gdept, NemoSEOSConfig())
    n2 = compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, NemoSEOSConfig(), g=NEMO_CONSTANTS_CONFIG.g,
        e3w_int=e3w, zrw_evaluation="nemo_literal",
        zrw_gdept_0=zrw_gdept_0, zrw_gdepw_0=zrw_gdepw_0,
        zrw_stretch=zrw_stretch)
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
        g=NEMO_CONSTANTS_CONFIG.g, e3w_int=e3w,
        zrw_evaluation="nemo_literal",
        zrw_gdept_0=zrw_gdept_0, zrw_gdepw_0=zrw_gdepw_0,
        zrw_stretch=zrw_stretch)
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
            from legoesm.ocean.experiments.dino import (
                dino_lat_lon_surface_forcing_arrays,
            )
            row8_utau_prod = metrics(
                np.asarray(forcing["tau_u_cell_2d"])[..., None],
                utau_n[..., None], wet2[..., None], focus, POINTWISE_BAR)
            # Exercise the ordinary production default without the harness's
            # explicit mesh donor.  This must reconstruct usrdef_hgr's raw
            # degree operand and match the same sbc dump at the registered bar.
            implicit_literal_forcing = dino_lat_lon_surface_forcing_arrays(
                br.geometry, cfg)
            row8_utau_default_literal = metrics(
                np.asarray(implicit_literal_forcing[
                    "tau_u_cell_2d"])[..., None],
                utau_n[..., None], wet2[..., None], focus, POINTWISE_BAR)
            row8_utau_literal = metrics(
                utau_literal[..., None], utau_n[..., None], wet2[..., None],
                focus, POINTWISE_BAR)
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
            row8m["default_literal_utau_vs_sbc_dump"] = (
                row8_utau_default_literal)
            row8m["legacy_factored_utau_vs_sbc_dump"] = row8_legacy_utau
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
                if not row8_utau_default_literal["pass"]:
                    raise AssertionError(
                        "row-8 default literal source construction did not "
                        "match sbc_dump_utau.bin")
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
                        "default_literal_utau_vs_dump":
                            row8_utau_default_literal,
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
                        twin_state.H_bathy.data,
                        evaluation="nemo_reciprocal"))
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
                    exceeded_n = pe_n[..., :-1] > half_n[..., None]
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
                    # Row 10 is the ENTIRE registered ln_lc operation through
                    # the line-463 en update, not merely its rn2b input.  The
                    # direct post-LC stream is therefore dispositive.  Fail
                    # closed on every path needed to prove the write-only
                    # bracket and source/binary provenance even though the
                    # later row-17 recurrence is not reached.
                    row10_provenance = (
                        args.row17_postlc_dump,
                        args.row17_bracket_off_restart,
                        args.row17_bracket_instrument_restart,
                        args.row17_instrument_source,
                        args.row17_instrument_binary,
                    )
                    if any(path is None for path in row10_provenance):
                        raise FileNotFoundError(
                            "row 10 requires the post-Langmuir dump plus "
                            "instrument source/binary and both bracket restarts")
                    for path in row10_provenance:
                        if not path.is_file():
                            raise FileNotFoundError(path)
                    bracket10 = restart_numeric_identity(
                        args.row17_bracket_instrument_restart,
                        args.row17_bracket_off_restart)
                    if not bracket10["pass"]:
                        raise AssertionError(
                            "row-10 post-LC dump-on/dump-off numeric restart "
                            "bracket failed")
                    if solve_kwargs.get("external_source") is None:
                        raise AssertionError(
                            "row 10 reached ln_lc=T without a solver source")
                    solver_source = np.asarray(
                        solve_kwargs["external_source"])[..., :nlc]
                    lc_source = np.asarray(lc_out)
                    # On the two DINO oracle cards the generic Veros EKE and
                    # bottom-dissipation source flags are disabled, so the
                    # solver source must be exactly the Langmuir source.  Do
                    # not silently generalize this association to a future
                    # mixed-source card.
                    if not np.array_equal(solver_source, lc_source):
                        raise AssertionError(
                            "DINO row 10 expected external_source to contain "
                            "only the captured Langmuir source")
                    dt10 = np.float64(solve_kwargs["dt"])
                    en_entry = np.asarray(solve_kwargs["e_old"])[..., :nlc]
                    literal_rhs = solve_kwargs.get("literal_external_rhs")
                    if literal_rhs is None:
                        raise AssertionError(
                            "row 10 literal source reached the solver without "
                            "the source-ordered line-463 en update")
                    postlc_l = np.asarray(literal_rhs)[..., :nlc]
                    rate_first_postlc = en_entry + dt10 * solver_source
                    postlc_n_full = base._load_interior(
                        str(args.row17_postlc_dump), ni, nj)
                    postlc_n = postlc_n_full[..., 1:1 + nlc]
                    row10_output = metrics(
                        postlc_l, postlc_n, wet_lc, focus, POINTWISE_BAR)
                    row10_inputs["langmuir_source_composite"] = row10m
                    row10_inputs["legacy_rate_first_line463_update"] = metrics(
                        rate_first_postlc, postlc_n, wet_lc, focus,
                        POINTWISE_BAR)
                    row10_inputs["literal_line463_update"] = metrics(
                        postlc_l, postlc_n, wet_lc, focus, POINTWISE_BAR)
                    row10_inputs["solver_source_equals_captured_langmuir"] = (
                        metrics(solver_source, lc_source, wet_lc, focus,
                                POINTWISE_BAR))
                    row10_output["operand_metrics"] = row10_inputs
                    row10_output["controls"] = planted_controls(
                        postlc_n, postlc_n, wet_lc, POINTWISE_BAR)
                    row10_output["instrumentation_bracket"] = bracket10
                    row10_output["instrumentation"] = {
                        "slot": "tke_dump_en_postlc.bin",
                        "time_level": time_level_for_dump(
                            "tke_dump_en_postlc.bin"),
                        "location": (
                            "immediately after active base zdftke.F90:401-468; "
                            "line 463 is the source output update"),
                        "sha256": {
                            str(path): sha256(path)
                            for path in row10_provenance
                        },
                        "resolved_evaluation":
                            lc_cfg.tke_langmuir_evaluation,
                    }
                    row4["continuation_preview"]["rows"][
                        "10_langmuir_source_output"] = row10_output
                    if not row10_output["pass"]:
                        row4["continuation_preview"]["first_divergence"] = {
                            "row": 10,
                            "operation": "Langmuir TKE source output",
                            "nemo_line":
                                "cfgs/DINO/MY_SRC/zdftke.F90:401-468 "
                                "(en update at :463)",
                            "output": row10_output,
                            "first_failing_operand": {
                                "name": "vectorized Langmuir source composite",
                                "nemo_line":
                                    "cfgs/DINO/MY_SRC/zdftke.F90:422-463",
                                "legoesm_line":
                                    "packages/ocean/legoesm/ocean/physics/"
                                    "vertical_mixing/tke.py:1809-1890",
                            },
                            "operand_localization": row10_inputs,
                            "next_round_fix_design": {
                                "status": "DESIGNED_NOT_IMPLEMENTED",
                                "option": "tke_langmuir_evaluation",
                                "faithful_default_on_complete_dino_nemo_cards":
                                    "nemo_literal",
                                "legacy_default_everywhere_else": "vectorized",
                                "legacy_opt_in_on_dino_nemo_cards":
                                    "vectorized",
                                "construction": (
                                    "Transcribe the active no-Stokes zWlc2 "
                                    "arm and zpelc top-down recurrence, then "
                                    "initialize imlc per column to mbkt+1 "
                                    "from carried bottom_level and perform "
                                    "NEMO's bottom-up overwrite scan. Continue "
                                    "with zhlc, zus/zus3, zwlc, and the line-"
                                    "463 en update in source order. This per-"
                                    "column fallback is required for unequal-"
                                    "depth no-crossing columns; a global "
                                    "deepest-interface fallback is not NEMO."),
                                "source_scope_guard": (
                                    "The two DINO oracle cards currently have "
                                    "generic EKE-recycling and bottom-"
                                    "dissipation sources disabled. Keep "
                                    "Langmuir separate from generic external "
                                    "sources, or fail closed, before enabling "
                                    "nemo_literal on any mixed-source card."),
                                "required_red_tests": [
                                    "hand-computed multi-level zpelc/imlc tie case",
                                    "planted first-versus-last threshold crossing",
                                    "unequal shallow/deep no-crossing columns "
                                    "using each column's mbkt+1 fallback",
                                    "direct post-LC dump census",
                                    "JIT and finite gradient",
                                    "unchanged-card byte identity",
                                ],
                                "reason_for_stop": (
                                    "The ordered cumulative/reverse-selection "
                                    "source path and its AD/JIT tests are a "
                                    "single fix too large for this round."),
                            },
                            "controls": row10_output["controls"],
                            "loud_retraction": (
                                "RETRACTED: round 9 initially marked only "
                                "row 10's rn2b operand VERIFIED and promoted "
                                "rows 11-14. The preregistered row is the full "
                                "Langmuir operation through line 463; its "
                                "direct output diverges, so rows 11-32 are "
                                "UNMEASURED under the ordered stop rule."),
                        }
                        later_operations = {
                            11: "Richardson and inverse-Prandtl assembly",
                            12: "TKE diffusion matrix",
                            13: "TKE RHS shear",
                            14: "TKE RHS stratification",
                            15: "TKE RHS dissipation",
                            16: "wave-coupled surface boundary",
                            17: "TKE implicit solve and floor/mask",
                            18: "nn_etau penetration",
                            19: "buoyancy mixing length",
                            20: "nn_mxl=3 limiting scans",
                            21: "base avm/avt/dissl assembly",
                            22: "inverse-Prandtl avt correction",
                            23: "closure coefficient copy",
                            24: "river-mouth enhancement",
                            25: "EVD tracer overwrite",
                            26: "EVD momentum overwrite",
                            27: "avs and disabled enhancements",
                            28: "turbocline scan and hmld",
                            29: "avm lateral boundary update",
                            30: "ldf_slp",
                            31: "momentum implicit solve application",
                            32: "tracer implicit solve application",
                        }
                        for number, operation in later_operations.items():
                            row4["continuation_preview"]["rows"][
                                f"{number}_ordered_stop"] = {
                                    "pass": False,
                                    "disposition": "UNMEASURED",
                                    "operation": operation,
                                    "reason": "ordered stop at row 10",
                                }
                    else:
                        # Row 11: literal nn_pdl=1 Richardson / inverse-
                        # Prandtl chain. The first production call is the
                        # pre-solve call using carried p_avm and frozen p_sh2;
                        # the second is tke_avn after the solve (row 22).
                        pr_args, pr_kwargs, pr_out = (
                            tke_capture["prandtl_calls"][0])
                        if pr_kwargs or len(pr_args) != 5:
                            raise AssertionError(
                                "unexpected _prandtl_number call contract")
                        rn2b_l, _shear_l, avm_l, pr_cfg, sh2_l = pr_args
                        npr = np.asarray(pr_out).shape[-1]
                        rn2b_l = np.asarray(rn2b_l)[..., :npr]
                        avm_l = np.asarray(avm_l)[..., :npr]
                        sh2_l = np.asarray(sh2_l)[..., :npr]
                        pdlr_l = 1.0 / np.asarray(pr_out)[..., :npr]
                        bshear = np.float64(pr_cfg.bshear_floor)
                        zdiv_l = sh2_l + bshear
                        safe_zdiv_l = np.where(zdiv_l == 0.0, 1.0, zdiv_l)
                        numerator_l = rn2b_l * avm_l
                        zri_strat_l = np.where(
                            zdiv_l == 0.0,
                            numerator_l / bshear,
                            numerator_l / safe_zdiv_l)
                        zri_l = np.where(rn2b_l <= 0.0, 0.0, zri_strat_l)

                        zri_n_full = base._load_interior(
                            str(RUN / "tke_dump_zri.bin"), ni, nj)
                        pdlr_n_full = base._load_interior(
                            str(RUN / "tke_dump_pdlr.bin"), ni, nj)
                        avm_n_full = base._load_interior(
                            str(RUN / "tke_dump_avm_in.bin"), ni, nj)
                        sh2_n_full = base._load_interior(
                            str(RUN / "tke_dump_sh2.bin"), ni, nj)
                        zri_n = zri_n_full[..., 1:1 + npr]
                        pdlr_n = pdlr_n_full[..., 1:1 + npr]
                        avm_n = avm_n_full[..., 1:1 + npr]
                        sh2_n_pr = sh2_n_full[..., 1:1 + npr]
                        rn2b_n_pr = n2_n_full[..., 1:1 + npr]
                        wet_pr = wet_w_all[..., :npr]

                        row11_inputs = {
                            "rn2b": metrics(
                                rn2b_l, rn2b_n_pr, wet_pr, focus,
                                POINTWISE_BAR),
                            "p_avm": metrics(
                                avm_l, avm_n, wet_pr, focus, POINTWISE_BAR),
                            "p_sh2": metrics(
                                sh2_l, sh2_n_pr, wet_pr, focus,
                                POINTWISE_BAR),
                            "zri": metrics(
                                zri_l, zri_n, wet_pr, focus, POINTWISE_BAR),
                            "p_pdlr": metrics(
                                pdlr_l, pdlr_n, wet_pr, focus,
                                POINTWISE_BAR),
                        }
                        def literal_zri(n2_operand, avm_operand, sh2_operand):
                            zdiv = sh2_operand + bshear
                            numerator = n2_operand * avm_operand
                            safe_zdiv = np.where(zdiv == 0.0, 1.0, zdiv)
                            stratified = np.where(
                                zdiv == 0.0,
                                numerator / bshear,
                                numerator / safe_zdiv)
                            return np.where(
                                n2_operand <= 0.0, 0.0, stratified)

                        substitution_arrays = {
                            "rn2b": literal_zri(
                                rn2b_n_pr, avm_l, sh2_l),
                            "p_avm": literal_zri(
                                rn2b_l, avm_n, sh2_l),
                            "p_sh2": literal_zri(
                                rn2b_l, avm_l, sh2_n_pr),
                            "all_three_oracle_operands": literal_zri(
                                rn2b_n_pr, avm_n, sh2_n_pr),
                        }
                        row11_substitutions = {
                            name: metrics(value, zri_n, wet_pr, focus,
                                          POINTWISE_BAR)
                            for name, value in substitution_arrays.items()
                        }
                        bn2_chain = None
                        if args.bn2_chain_dir is not None:
                            chain_files = {
                                name: args.bn2_chain_dir / f"bn2_dump_{name}.bin"
                                for name in (
                                    "zrw", "zaw", "zbw", "numerator", "result")
                            }
                            missing = [str(path) for path in chain_files.values()
                                       if not path.is_file()]
                            if missing:
                                raise FileNotFoundError(
                                    f"missing registered row-11 dumps: {missing}")
                            chain_n_full = {
                                name: base._load_haloed(str(path), jpi, jpj, hls)
                                for name, path in chain_files.items()
                            }
                            chain_n = {
                                name: value[..., 1:1 + npr]
                                for name, value in chain_n_full.items()
                            }
                            if not np.array_equal(
                                    chain_n["result"], rn2b_n_pr,
                                    equal_nan=True):
                                raise AssertionError(
                                    "bn2 result stream is not bit-identical to "
                                    "the independent tke_dump_rn2b time-level pin")

                            gd_up = np.asarray(gdept)[..., :npr]
                            gd_lo = np.asarray(gdept)[..., 1:1 + npr]
                            gw_l = np.asarray(gdepw)[..., :npr]
                            aa_up = np.asarray(alpha)[..., :npr]
                            aa_lo = np.asarray(alpha)[..., 1:1 + npr]
                            bb_up = np.asarray(beta)[..., :npr]
                            bb_lo = np.asarray(beta)[..., 1:1 + npr]
                            e3_l = np.asarray(e3w)[..., :npr]
                            t_up = np.asarray(T)[..., :npr]
                            t_lo = np.asarray(T)[..., 1:1 + npr]
                            s_up = np.asarray(S)[..., :npr]
                            s_lo = np.asarray(S)[..., 1:1 + npr]
                            zrw_l = np.asarray(_nemo_bn2_zrw(
                                jnp.asarray(np.asarray(gdept)[..., :npr + 1]),
                                jnp.asarray(gw_l), evaluation="nemo_literal",
                                gdept_0=jnp.asarray(
                                    np.asarray(zrw_gdept_0)[..., :npr + 1]),
                                gdepw_0=jnp.asarray(
                                    np.asarray(zrw_gdepw_0)[..., :npr]),
                                stretch=jnp.asarray(zrw_stretch)))
                            zaw_l = aa_lo * (1.0 - zrw_l) + aa_up * zrw_l
                            zbw_l = bb_lo * (1.0 - zrw_l) + bb_up * zrw_l
                            num_l = np.float64(NEMO_CONSTANTS_CONFIG.g) * (
                                zaw_l * (t_up - t_lo)
                                - zbw_l * (s_up - s_lo))
                            result_l = num_l / e3_l
                            chain_l = {
                                "zrw": zrw_l,
                                "zaw": zaw_l,
                                "zbw": zbw_l,
                                "numerator": num_l,
                                "result": result_l,
                            }
                            checkpoints = {}
                            for name in (
                                    "zrw", "zaw", "zbw", "numerator", "result"):
                                checkpoint = metrics(
                                    chain_l[name], chain_n[name], wet_pr,
                                    focus, POINTWISE_BAR)
                                checkpoint["exact"] = exact_difference_census(
                                    chain_l[name], chain_n[name], wet_pr, focus)
                                checkpoint["controls"] = planted_controls(
                                    chain_n[name], chain_n[name], wet_pr,
                                    POINTWISE_BAR)
                                checkpoints[name] = checkpoint

                            # Suffix substitutions answer which earliest NEMO
                            # checkpoint is sufficient to close the amplified
                            # Richardson zri failure.
                            zaw_from_nemo_zrw = (
                                aa_lo * (1.0 - chain_n["zrw"])
                                + aa_up * chain_n["zrw"])
                            zbw_from_nemo_zrw = (
                                bb_lo * (1.0 - chain_n["zrw"])
                                + bb_up * chain_n["zrw"])
                            result_from_zrw = (
                                np.float64(NEMO_CONSTANTS_CONFIG.g) * (
                                    zaw_from_nemo_zrw * (t_up - t_lo)
                                    - zbw_from_nemo_zrw * (s_up - s_lo)) / e3_l)
                            result_from_zaw_zbw = (
                                np.float64(NEMO_CONSTANTS_CONFIG.g) * (
                                    chain_n["zaw"] * (t_up - t_lo)
                                    - chain_n["zbw"] * (s_up - s_lo)) / e3_l)
                            result_from_numerator = chain_n["numerator"] / e3_l
                            suffix_results = {
                                "zrw": result_from_zrw,
                                "zaw_zbw": result_from_zaw_zbw,
                                "numerator": result_from_numerator,
                                "result": chain_n["result"],
                            }
                            zri_suffix = {
                                name: metrics(
                                    literal_zri(value, avm_l, sh2_l),
                                    zri_n, wet_pr, focus, POINTWISE_BAR)
                                for name, value in suffix_results.items()
                            }
                            first_exact = next(
                                (name for name in (
                                    "zrw", "zaw", "zbw", "numerator", "result")
                                 if not checkpoints[name]["exact"]["bit_identical"]),
                                None)
                            first_closing = next(
                                (name for name in (
                                    "zrw", "zaw_zbw", "numerator", "result")
                                 if zri_suffix[name]["pass"]), None)

                            bracket = restart_numeric_identity(
                                args.bn2_bracket_instrument_restart,
                                args.bn2_bracket_off_restart)
                            if not bracket["pass"]:
                                raise AssertionError(
                                    "dump-on/dump-off numeric restart bracket failed")
                            certified = restart_numeric_identity(
                                args.bn2_bracket_instrument_restart,
                                args.bn2_certified_restart)
                            with xr.open_dataset(
                                    args.bn2_bracket_instrument_restart,
                                    decode_times=False) as ds:
                                planted_left = np.asarray(ds["tb"])
                                planted_right = np.asarray(ds["sb"])
                            planted_wrong_field_fires = not np.array_equal(
                                planted_left, planted_right, equal_nan=True)
                            if not planted_wrong_field_fires:
                                raise AssertionError(
                                    "restart bracket planted wrong-field control did not fire")
                            bn2_chain = {
                                "nemo_line": "src/OCE/TRA/eosbn2.F90:1459-1468",
                                "ordered_checkpoints": checkpoints,
                                "first_bitwise_diverging_checkpoint": first_exact,
                                "zri_suffix_substitutions": zri_suffix,
                                "first_zri_closing_substitution": first_closing,
                                "result_stream_vs_tke_rn2b_bit_identical": True,
                                "instrumentation_bracket": bracket,
                                "certified_comparison": certified,
                                "certified_all_numeric_identity_claim_retracted": (
                                    not certified["pass"]),
                                "certified_retraction_reason": (
                                    "the sandbox already carries the registered "
                                    "literal tau diagnostic, whereas the older "
                                    "certified binary does not; only the same-source "
                                    "dump-off build is a valid instrumentation bracket"
                                    if not certified["pass"] else None),
                                "planted_wrong_field_control_fires": (
                                    planted_wrong_field_fires),
                                "sha256": {
                                    str(path): sha256(path)
                                    for path in [
                                        *chain_files.values(),
                                        args.bn2_bracket_off_restart,
                                        args.bn2_bracket_instrument_restart,
                                        args.bn2_certified_restart,
                                    ]
                                },
                            }
                        row11_primary = dict(
                            row11_inputs["zri"]
                            if not row11_inputs["zri"]["pass"]
                            else row11_inputs["p_pdlr"])
                        row11_primary["operand_metrics"] = row11_inputs
                        row11_primary["single_operand_substitutions"] = (
                            row11_substitutions)
                        row11_primary["first_passing_substitution"] = next(
                            (name for name, value in
                             row11_substitutions.items() if value["pass"]),
                            None)
                        row11_primary["ownership_verdict"] = {
                            "owner": "rn2b",
                            "statement": (
                                "The row-10 rn2b residual passes its own "
                                "1e-15 column bar but Richardson division "
                                "amplifies it to the row-11 zri failures; "
                                "substituting only NEMO rn2b closes zri "
                                "exactly, while substituting p_avm or p_sh2 "
                                "does not change the failing-column count."),
                            "rn2b_row10_bad_columns": 0,
                            "zri_row11_bad_columns": int(
                                row11_inputs["zri"]["n_diverged_columns"]),
                        }
                        row11_primary["next_round_fix_design"] = {
                            "status": "DESIGNED_NOT_IMPLEMENTED",
                            "reason_for_stop": (
                                "The correction requires new NEMO internal "
                                "dump slots and a bracket-stream identity run; "
                                "it is too large for this round."),
                            "nemo_source_order": [
                                "zrw", "zaw", "zbw",
                                "thermal_minus_haline_numerator",
                                "raw_divide_by_e3w_result",
                            ],
                            "nemo_line": "src/OCE/TRA/eosbn2.F90:1459-1468",
                            "instrumentation": (
                                "Extend the existing write-only MY_SRC dump "
                                "pattern for each listed intermediate, run the "
                                "one-step CPU oracle, and prove the pre/post "
                                "instrumentation bracket stream bit-identical."),
                            "lego_walk": (
                                "Capture the matching step-entry bundle "
                                "operands and stop at the first failed slot."),
                            "production_selection": (
                                "If literal association is the owner, select "
                                "nemo_literal only under step_entry; preserve "
                                "implicit_solve_state byte-for-byte."),
                            "required_controls": [
                                "planted_scalar_perturbation",
                                "one_i_horizontal_roll", "nonfinite_poison",
                                "instrumented_bracket_stream_identity",
                            ],
                        }
                        row11_primary["resolved_constants"] = {
                            "rn_bshear": float(bshear),
                            "ri_cri": float(1.0 / pr_cfg.prandtl_ri_coeff),
                            "tke_n2_evaluation_stage": (
                                pr_cfg.tke_n2_evaluation_stage),
                        }
                        row11_primary["controls"] = {
                            "zri": planted_controls(
                                zri_n, zri_n, wet_pr, POINTWISE_BAR),
                            "p_pdlr": planted_controls(
                                pdlr_n, pdlr_n, wet_pr, POINTWISE_BAR),
                        }
                        if bn2_chain is not None:
                            row11_primary["bn2_operand_chain"] = bn2_chain
                        row4["continuation_preview"]["rows"][
                            "11_prandtl_zri_pdlr"] = row11_primary
                        if not row11_primary["pass"]:
                            first_name = next(
                                name for name in (
                                    "rn2b", "p_avm", "p_sh2", "zri",
                                    "p_pdlr")
                                if not row11_inputs[name]["pass"])
                            row4["continuation_preview"][
                                "first_divergence"] = {
                                    "row": 11,
                                    "operation": (
                                        "Richardson and inverse-Prandtl "
                                        "assembly"),
                                    "nemo_line": (
                                        "cfgs/DINO/MY_SRC/zdftke.F90:"
                                        "477-496"),
                                    "output": row11_primary,
                                    "first_failing_operand": {
                                        "name": first_name,
                                        "nemo_line": (
                                            "cfgs/DINO/MY_SRC/zdftke.F90:"
                                            "480-495"),
                                    },
                                    "operand_localization": row11_inputs,
                                    "controls": row11_primary["controls"],
                                }
                        else:
                            # Row 12: compare the production matrix actually
                            # handed to Thomas against zdftke.F90:499-510,
                            # reconstructed from independent dumped operands.
                            tri_calls = tke_capture["tri_calls"]
                            literal_tri_calls = tke_capture[
                                "literal_tri_calls"]
                            if tri_calls:
                                tri_args, _tri_out = tri_calls[0]
                            elif literal_tri_calls:
                                literal_args, _literal_kwargs, _tri_out = (
                                    literal_tri_calls[0])
                                tri_args = literal_args[:4]
                            else:
                                raise AssertionError(
                                    "TKE tridiagonal solve was not captured")
                            a_tri, b_tri, c_tri, _rhs_tri = map(
                                np.asarray, tri_args)
                            nmat = npr - 1
                            wet_mat = wet_pr[..., :nmat]
                            if a_tri.shape[-1] != npr + 1:
                                raise AssertionError(
                                    "row-12 expected virtual surface + 35 "
                                    f"TKE rows, got {a_tri.shape[-1]}")
                            # Drop the virtual surface identity row. The next
                            # 34 rows are Fortran jk=2..jpkm1.
                            lw_l = a_tri[..., 1:1 + nmat]
                            diag_l = b_tri[..., 1:1 + nmat]
                            up_l = c_tri[..., 1:1 + nmat]
                            solve_kwargs = tke_capture["solve_calls"][0][0]

                            with xr.open_dataset(
                                    RUN / "mesh_mask.nc",
                                    decode_times=False) as ds:
                                e3t0_m = np.moveaxis(np.asarray(
                                    ds["e3t_0"].isel(time_counter=0)), 0, -1)
                                tmask_m = np.moveaxis(np.asarray(
                                    ds["tmask"].isel(time_counter=0)), 0, -1)
                            stretch_m = np.asarray(zrw_stretch)
                            e3t_m = e3t0_m * stretch_m[..., None]
                            e3w_m = np.asarray(e3w)[..., :nmat]
                            avm_full = avm_n_full
                            dissl_full = base._load_interior(
                                str(RUN / "tke_dump_dissl.bin"), ni, nj)
                            avm_jk = avm_full[..., 1:1 + nmat]
                            avm_up = avm_full[..., 2:2 + nmat]
                            avm_lw = avm_full[..., :nmat]
                            e3t_jk = e3t_m[..., 1:1 + nmat]
                            e3t_lw = e3t_m[..., :nmat]
                            tmask_jk = tmask_m[..., 1:1 + nmat]
                            dissl_jk = dissl_full[..., 1:1 + nmat]
                            wmask_jk = wet_mat.astype(np.float64)
                            # Score the timestep actually handed to zdftke.
                            # NEMO's TKE equation uses base rn_Dt=2700 even
                            # while MLF tracer/momentum solves use
                            # rDt=2*rn_Dt (dom_oce.F90:69-71;
                            # MY_SRC/zdftke.F90:336-337).
                            dt64 = np.float64(solve_kwargs["dt"])
                            ediss64 = np.float64(tke_cfg.c_eps)
                            zcof_n = (-np.float64(0.5) * dt64) * tmask_jk
                            avm_up_sum_n = np.maximum(
                                avm_up + avm_jk, np.float64(2.0e-5))
                            avm_lw_sum_n = np.maximum(
                                avm_jk + avm_lw, np.float64(2.0e-5))
                            up_n = (zcof_n * avm_up_sum_n
                                    / (e3t_jk * e3w_m))
                            lw_n = (zcof_n * avm_lw_sum_n
                                    / (e3t_lw * e3w_m))
                            diag_n = (np.float64(1.0) - lw_n - up_n
                                      + (np.float64(1.5) * dt64 * ediss64)
                                      * dissl_jk * wmask_jk)

                            avm_old_l = np.asarray(solve_kwargs["K_M_old"])
                            avm_operand_l = avm_old_l[..., :nmat]
                            avm_up_sum_l = np.maximum(
                                avm_old_l[..., 1:1 + nmat]
                                + avm_old_l[..., :nmat],
                                np.float64(2.0e-5))
                            avm_surface_l = np.asarray(
                                solve_kwargs["K_M_surface"])
                            avm_lw_sum_l = np.maximum(
                                np.concatenate([
                                    avm_surface_l[..., None]
                                    + avm_old_l[..., :1],
                                    (avm_old_l[..., 1:nmat]
                                     + avm_old_l[..., :nmat - 1]),
                                ], axis=-1),
                                np.float64(2.0e-5))
                            zcof_l = (
                                -np.float64(0.5) * dt64
                                * np.asarray(
                                    solve_kwargs["w_active"])[..., :nmat])
                            incoming_e3w_l = np.asarray(
                                solve_kwargs["dz_half"])[..., :nmat]
                            nemo_e3t_operand = solve_kwargs.get("nemo_e3t")
                            if nemo_e3t_operand is not None:
                                nemo_e3t_operand = np.asarray(nemo_e3t_operand)
                                up_e3t_operand_l = (
                                    nemo_e3t_operand[..., 1:1 + nmat])
                                lw_e3t_operand_l = nemo_e3t_operand[..., :nmat]
                                effective_e3w_l = incoming_e3w_l
                                metric_operand_source = (
                                    "literal live nemo_e3t and step-entry e3w")
                            elif solve_kwargs["dz_cell"] is None:
                                # Legacy assembly feeds e3w into the face-
                                # gradient slot where NEMO reads e3t. Its
                                # control-volume slot is the first e3w row
                                # repeated, then the preceding e3w row.
                                dz_legacy = np.asarray(
                                    solve_kwargs["dz_half"])
                                up_e3t_operand_l = dz_legacy[..., :nmat]
                                effective_e3w_l = np.concatenate(
                                    [dz_legacy[..., :1],
                                     dz_legacy[..., :nmat - 1]], axis=-1)
                                lw_e3t_operand_l = np.concatenate(
                                    [np.asarray(solve_kwargs["dz_surface"])
                                     [..., None],
                                     dz_legacy[..., :nmat - 1]], axis=-1)
                                metric_operand_source = (
                                    "legacy dz_face/dz_int_eff; dz_cell=None")
                            else:
                                up_e3t_operand_l = np.asarray(
                                    solve_kwargs["dz_cell"])[..., 1:1 + nmat]
                                lw_e3t_operand_l = np.asarray(
                                    solve_kwargs["dz_cell"])[..., :nmat]
                                effective_e3w_l = incoming_e3w_l
                                metric_operand_source = (
                                    "live dz_cell and dz_half")
                            if solve_kwargs.get("dissl_old") is not None:
                                dissl_operand_l = np.asarray(
                                    solve_kwargs["dissl_old"])[..., :nmat]
                                dissl_operand_source = (
                                    "restart/previous-step carried dissl")
                            else:
                                dissl_operand_l = (
                                    np.sqrt(np.maximum(
                                        np.asarray(solve_kwargs["e_old"])
                                        [..., :nmat],
                                        np.float64(tke_cfg.tke_background)))
                                    / np.maximum(
                                        np.asarray(solve_kwargs["l_eps"])
                                        [..., :nmat],
                                        np.float64(tke_mod._mixing_length_floor(
                                            tke_cfg))))
                                dissl_operand_source = "factored sqrt(en)/zmxld"
                            row12_operands = {
                                "zcof_tmask": metrics(
                                    zcof_l, zcof_n, wet_mat, focus,
                                    POINTWISE_BAR),
                                "p_avm_jk": metrics(
                                    avm_operand_l, avm_jk, wet_mat, focus,
                                    POINTWISE_BAR),
                                "up_avm_sum": metrics(
                                    avm_up_sum_l, avm_up_sum_n, wet_mat,
                                    focus, POINTWISE_BAR),
                                "lower_avm_sum": metrics(
                                    avm_lw_sum_l, avm_lw_sum_n, wet_mat,
                                    focus, POINTWISE_BAR),
                                "up_e3t_jk_Kmm": metrics(
                                    up_e3t_operand_l, e3t_jk, wet_mat, focus,
                                    POINTWISE_BAR),
                                "lower_e3t_jkm1_Kmm": metrics(
                                    lw_e3t_operand_l, e3t_lw, wet_mat, focus,
                                    POINTWISE_BAR),
                                "effective_e3w_jk_Kmm": metrics(
                                    effective_e3w_l, e3w_m, wet_mat, focus,
                                    POINTWISE_BAR),
                                "incoming_e3w_Kmm_control": metrics(
                                    incoming_e3w_l, e3w_m, wet_mat, focus,
                                    POINTWISE_BAR),
                                "dissl_jk": metrics(
                                    dissl_operand_l, dissl_jk, wet_mat, focus,
                                    POINTWISE_BAR),
                            }
                            row12_coefficients = {
                                "zd_up": metrics(
                                    up_l, up_n, wet_mat, focus, POINTWISE_BAR),
                                "zd_lw": metrics(
                                    lw_l, lw_n, wet_mat, focus, POINTWISE_BAR),
                                "zdiag": metrics(
                                    diag_l, diag_n, wet_mat, focus,
                                    POINTWISE_BAR),
                            }
                            first_coeff = next(
                                (name for name in ("zd_up", "zd_lw", "zdiag")
                                 if not row12_coefficients[name]["pass"]),
                                None)
                            primary_name = first_coeff or "zdiag"
                            row12_primary = dict(
                                row12_coefficients[primary_name])
                            row12_primary["coefficient_metrics"] = (
                                row12_coefficients)
                            row12_primary["operand_metrics"] = row12_operands
                            row12_primary["resolved_constants"] = {
                                "rn_Dt": float(dt64),
                                "expected_nemo_rn_Dt": float(kamm.DT),
                                "rn_ediss": float(ediss64),
                                "minimum_avm_sum": 2.0e-5,
                                "production_metric_operand_source":
                                    metric_operand_source,
                                "production_dissl_operand_source":
                                    dissl_operand_source,
                                "loud_retraction": (
                                    "RETRACTED: the first round-9 run called "
                                    "the literal coefficient association the "
                                    "owner while reconstructing zcof with "
                                    "2700 s but failing to score the solver's "
                                    "consumed 5400 s MLF dt. The corrected "
                                    "probe scores solve_kwargs['dt']; the "
                                    "production MLF path now threads NEMO's "
                                    "base rn_Dt only to nemo_literal TKE."),
                            }
                            row12_primary["controls"] = {
                                name: planted_controls(
                                    ref, ref, wet_mat, POINTWISE_BAR)
                                for name, ref in (
                                    ("zd_up", up_n), ("zd_lw", lw_n),
                                    ("zdiag", diag_n))
                            }
                            row4["continuation_preview"]["rows"][
                                "12_tke_diffusion_matrix"] = row12_primary
                            if first_coeff is not None:
                                coefficient_operands = {
                                    "zd_up": (
                                        "zcof_tmask", "up_avm_sum",
                                        "up_e3t_jk_Kmm",
                                        "effective_e3w_jk_Kmm"),
                                    "zd_lw": (
                                        "zcof_tmask", "lower_avm_sum",
                                        "lower_e3t_jkm1_Kmm",
                                        "effective_e3w_jk_Kmm"),
                                    "zdiag": ("dissl_jk",),
                                }
                                first_operand = next(
                                    (name for name in
                                     coefficient_operands[first_coeff]
                                     if not row12_operands[name]["pass"]),
                                    "literal coefficient evaluation order")
                                row4["continuation_preview"][
                                    "first_divergence"] = {
                                        "row": 12,
                                        "operation": (
                                            "TKE diffusion matrix lower/upper/"
                                            "diagonal coefficients"),
                                        "nemo_line": (
                                            "cfgs/DINO/MY_SRC/zdftke.F90:"
                                            "499-510"),
                                        "output": row12_primary,
                                        "first_failing_operand": {
                                            "name": first_operand,
                                            "first_failing_coefficient":
                                                first_coeff,
                                            "nemo_line": (
                                                "cfgs/DINO/MY_SRC/zdftke.F90:"
                                                "500-510"),
                                        },
                                        "operand_localization": row12_operands,
                                        "controls": row12_primary["controls"],
                                        "next_round_fix_design": {
                                            "option":
                                                "tke_matrix_evaluation",
                                            "faithful_default_on_complete_"
                                            "dino_nemo_cards": "nemo_literal",
                                            "legacy_default_everywhere_else":
                                                "factored",
                                            "legacy_opt_in_on_dino_nemo_cards":
                                                "factored",
                                            "construction": (
                                                "assemble zcof, zd_up, zd_lw "
                                                "and zdiag in zdftke.F90 "
                                                "source order from carried "
                                                "p_avm/dissl and live Kmm "
                                                "metrics before calling the "
                                                "same differentiable Thomas "
                                                "solver"),
                                        },
                                    }
                            else:
                                # Rows 13-15 are the three source-ordered RHS
                                # operands at zdftke.F90:513-515.  Score the
                                # exact arrays consumed by the solve, not a
                                # recomputed post-solve diagnostic.
                                nrhs = npr
                                wet_rhs = wet_pr[..., :nrhs]
                                p_sh2_l = np.asarray(
                                    solve_kwargs["P_s"])[..., :nrhs]
                                p_sh2_n = sh2_n_full[..., 1:1 + nrhs]
                                row13m = metrics(
                                    p_sh2_l, p_sh2_n, wet_rhs, focus,
                                    POINTWISE_BAR)
                                row13m["controls"] = planted_controls(
                                    p_sh2_n, p_sh2_n, wet_rhs, POINTWISE_BAR)
                                row4["continuation_preview"]["rows"][
                                    "13_tke_rhs_shear"] = row13m

                                restart_path = (
                                    RUN / "DINO_00005760_restart.nc")
                                _ravm, ravt, _rdissl = (
                                    kamm.read_nemo_restart_tke_coefficients(
                                        str(restart_path), nn_hls=0))
                                rn2_l = np.asarray(
                                    solve_kwargs["N2"])[..., :nrhs]
                                rn2_n_rhs = n2_now_n_full[..., 1:1 + nrhs]
                                avt_l = np.asarray(
                                    solve_kwargs["K_H_old"])[..., :nrhs]
                                avt_n = np.asarray(ravt)[..., 1:1 + nrhs]
                                strat_l = -(avt_l * rn2_l)
                                strat_n = -(avt_n * rn2_n_rhs)
                                row14m = metrics(
                                    strat_l, strat_n, wet_rhs, focus,
                                    POINTWISE_BAR)
                                row14m["operand_metrics"] = {
                                    "p_avt": metrics(
                                        avt_l, avt_n, wet_rhs, focus,
                                        POINTWISE_BAR),
                                    "rn2": metrics(
                                        rn2_l, rn2_n_rhs, wet_rhs, focus,
                                        POINTWISE_BAR),
                                }
                                row14m["controls"] = planted_controls(
                                    strat_n, strat_n, wet_rhs, POINTWISE_BAR)
                                row4["continuation_preview"]["rows"][
                                    "14_tke_rhs_stratification"] = row14m

                                restart_en = kamm.read_nemo_restart_en(
                                    str(restart_path), nn_hls=0)
                                en_l = np.asarray(
                                    solve_kwargs["e_old"])[..., :nrhs]
                                en_n = np.asarray(
                                    restart_en)[..., 1:1 + nrhs]
                                dissl_l = np.asarray(
                                    solve_kwargs["dissl_old"])[..., :nrhs]
                                dissl_n = np.asarray(
                                    _rdissl)[..., 1:1 + nrhs]
                                if (args.row17_postlc_dump is None
                                        or not args.row17_postlc_dump.is_file()):
                                    raise FileNotFoundError(
                                        "row 15 requires --row17-postlc-dump "
                                        "because zdftke.F90:546 consumes the "
                                        "post-Langmuir en")
                                postlc_n_full = base._load_interior(
                                    str(args.row17_postlc_dump), ni, nj)
                                postlc_n_int = postlc_n_full[
                                    ..., 1:1 + nrhs]
                                literal_rhs_l = solve_kwargs.get(
                                    "literal_external_rhs")
                                if literal_rhs_l is None:
                                    raise AssertionError(
                                        "row 15 requires production's literal "
                                        "line-463 post-Langmuir en operand")
                                postlc_l = np.asarray(
                                    literal_rhs_l)[..., :nrhs]
                                zfact3 = np.float64(0.5) * ediss64
                                diss_rhs_l = zfact3 * dissl_l * postlc_l
                                diss_rhs_n = zfact3 * dissl_n * postlc_n_int
                                row15m = metrics(
                                    diss_rhs_l, diss_rhs_n, wet_rhs, focus,
                                    POINTWISE_BAR)
                                row15m["operand_metrics"] = {
                                    "carried_dissl": metrics(
                                        dissl_l, dissl_n, wet_rhs, focus,
                                        POINTWISE_BAR),
                                    "post_langmuir_en": metrics(
                                        postlc_l, postlc_n_int, wet_rhs, focus,
                                        POINTWISE_BAR),
                                }
                                row15m["controls"] = planted_controls(
                                    diss_rhs_n, diss_rhs_n, wet_rhs,
                                    POINTWISE_BAR)
                                row4["continuation_preview"]["rows"][
                                    "15_tke_rhs_dissipation"] = row15m
                                row4["continuation_preview"]["rows"][
                                    "16_wave_surface_boundary"] = {
                                        "pass": True,
                                        "disposition": "WAIVED",
                                        "reason": (
                                            "Resolved DINO run has "
                                            "ln_wave=F; zdftke.F90:524-545 "
                                            "is not executed."),
                                        "nemo_line": (
                                            "cfgs/DINO/MY_SRC/zdftke.F90:"
                                            "524-545"),
                                    }
                                for row_number, operation, output, operands in (
                                    (13, "TKE RHS shear operand", row13m,
                                     {"p_sh2": row13m}),
                                    (14, "TKE RHS stratification operand",
                                     row14m, row14m["operand_metrics"]),
                                    (15, "TKE RHS dissipation operand",
                                     row15m, row15m["operand_metrics"]),
                                ):
                                    if not output["pass"]:
                                        first_operand = next(
                                            name for name, result in
                                            operands.items()
                                            if not result["pass"])
                                        row4["continuation_preview"][
                                            "first_divergence"] = {
                                                "row": row_number,
                                                "operation": operation,
                                                "nemo_line": (
                                                    "cfgs/DINO/MY_SRC/"
                                                    f"zdftke.F90:{500 + row_number}"),
                                                "output": output,
                                                "first_failing_operand": {
                                                    "name": first_operand},
                                                "operand_localization": operands,
                                                "controls": output["controls"],
                                            }
                                        if row_number == 15:
                                            row4["continuation_preview"][
                                                "first_divergence"].update({
                                                "loud_retraction": (
                                                    "RETRACTED: the earlier "
                                                    "row-15 scorer multiplied "
                                                    "carried dissl by step-entry "
                                                    "en. zdftke.F90:463 first "
                                                    "adds Langmuir energy and "
                                                    ":515 consumes that updated "
                                                    "en in the dissipation "
                                                    "add-back. The corrected "
                                                    "scorer uses the direct "
                                                    "post-Langmuir dump."),
                                                "next_round_fix_design": {
                                                    "status":
                                                        "DESIGNED_NOT_IMPLEMENTED",
                                                    "option":
                                                        "tke_langmuir_evaluation",
                                                    "faithful_default_on_complete_"
                                                    "dino_nemo_cards":
                                                        "nemo_literal",
                                                    "legacy_default_everywhere_"
                                                    "else": "vectorized",
                                                    "legacy_opt_in_on_dino_nemo_"
                                                    "cards": "vectorized",
                                                    "construction": (
                                                        "Transcribe zWlc2, the "
                                                        "top-down zpelc recurrence, "
                                                        "bottom-up imlc selection, "
                                                        "zhlc, zus/zus3, zwlc and "
                                                        "the en update at "
                                                        "zdftke.F90:422-463 as "
                                                        "ordered JAX scans; pass "
                                                        "the resulting post-LC en "
                                                        "unchanged into the line-"
                                                        "515 dissipation operand."),
                                                    "required_red_tests": [
                                                        "hand-computed multi-level "
                                                        "zpelc/imlc tie case",
                                                        "planted first-vs-last "
                                                        "threshold crossing",
                                                        "direct post-LC dump census",
                                                        "JIT and finite gradient",
                                                        "unchanged-card byte identity",
                                                    ],
                                                    "reason_for_stop": (
                                                        "The literal cumulative/"
                                                        "reverse-selection source "
                                                        "path and its AD/JIT tests "
                                                        "are a single fix too large "
                                                        "for the remainder of this "
                                                        "round."),
                                                },
                                            })
                                        break
                                if "first_divergence" not in row4[
                                        "continuation_preview"]:
                                    row17_args = (
                                        args.row17_postsolve_dump,
                                        args.row17_forward_diag_dump,
                                        args.row17_forward_rhs_dump,
                                        args.row17_pre_diag_dump,
                                        args.row17_pre_lower_dump,
                                        args.row17_pre_rhs_dump,
                                        args.row17_postlc_dump,
                                        args.row17_bracket_off_restart,
                                        args.row17_bracket_instrument_restart,
                                        args.row17_instrument_source,
                                        args.row17_instrument_binary,
                                    )
                                    if any(x is None for x in row17_args):
                                        row17m = {
                                            "pass": False,
                                            "disposition": "UNMEASURED",
                                            "reason": (
                                                "Row 17 requires all five "
                                                "--row17-* provenance paths."),
                                        }
                                    else:
                                        for path in row17_args:
                                            if not path.is_file():
                                                raise FileNotFoundError(path)
                                        postsolve_n_full = base._load_interior(
                                            str(args.row17_postsolve_dump),
                                            ni, nj)
                                        postsolve_n = postsolve_n_full[
                                            ..., 1:1 + nrhs]
                                        postsolve_l = np.asarray(
                                            tke_capture["solve_calls"][0][1])[
                                                ..., :nrhs]
                                        rhs_l = np.asarray(
                                            tri_args[3])[..., 1:1 + nrhs]
                                        en_n_with_bc = np.array(en_n, copy=True)
                                        np.put_along_axis(
                                            en_n_with_bc,
                                            bottom_level[..., None],
                                            bottom_prod[..., None], axis=-1)
                                        # The direct dump is NEMO's exact
                                        # post-line-463 en, including its
                                        # source association and boundary
                                        # state. It is the row-17 RHS base.
                                        rhs_n = postlc_n_full[
                                            ..., 1:1 + nrhs]
                                        rhs_n = rhs_n + dt64 * (
                                            p_sh2_n + strat_n + diss_rhs_n
                                        ) * wet_rhs.astype(np.float64)
                                        np.put_along_axis(
                                            rhs_n,
                                            bottom_level[..., None],
                                            bottom_prod[..., None], axis=-1)
                                        rhs_metric = metrics(
                                            rhs_l, rhs_n, wet_rhs, focus,
                                            POINTWISE_BAR)
                                        row17m = metrics(
                                            postsolve_l, postsolve_n, wet_rhs,
                                            focus, 1.0e-12)
                                        literal_recurrence = (
                                            nemo_literal_tke_recurrence(
                                                a_tri, b_tri, c_tri,
                                                tri_args[3], surface_n,
                                                np.asarray(
                                                    solve_kwargs["w_active"]),
                                                tke_cfg.tke_background))
                                        literal_diag, literal_work = (
                                            nemo_literal_tke_forward_stages(
                                                a_tri, b_tri, c_tri,
                                                tri_args[3], surface_n))
                                        diag_n = base._load_interior(
                                            str(args.row17_forward_diag_dump),
                                            ni, nj)
                                        work_n = base._load_interior(
                                            str(args.row17_forward_rhs_dump),
                                            ni, nj)
                                        pre_diag_n = base._load_interior(
                                            str(args.row17_pre_diag_dump),
                                            ni, nj)
                                        pre_lw_n = base._load_interior(
                                            str(args.row17_pre_lower_dump),
                                            ni, nj)
                                        pre_rhs_n = base._load_interior(
                                            str(args.row17_pre_rhs_dump),
                                            ni, nj)
                                        postlc_n = postlc_n_full
                                        n_forward = literal_diag.shape[-1] - 1
                                        wet_forward = np.concatenate(
                                            (wet2[..., None],
                                             wet_rhs[..., :n_forward - 1]),
                                            axis=-1)
                                        forward_diag_metric = metrics(
                                            literal_diag[..., :n_forward],
                                            diag_n[..., :n_forward],
                                            wet_forward, focus,
                                            POINTWISE_BAR)
                                        forward_rhs_metric = metrics(
                                            literal_work[..., :n_forward],
                                            work_n[..., :n_forward],
                                            wet_forward, focus,
                                            POINTWISE_BAR)
                                        pre_diag_l = np.array(
                                            b_tri, copy=True)
                                        pre_diag_l[..., 0] = (
                                            1.0 / np.asarray(surface_n))
                                        pre_lw_l = np.array(a_tri, copy=True)
                                        pre_lw_l[..., 0] = 1.0
                                        pre_stage_metrics = {
                                            "zdiag": metrics(
                                                pre_diag_l[..., :n_forward],
                                                pre_diag_n[..., :n_forward],
                                                wet_forward, focus,
                                                POINTWISE_BAR),
                                            "zd_lw": metrics(
                                                pre_lw_l[..., :n_forward],
                                                pre_lw_n[..., :n_forward],
                                                wet_forward, focus,
                                                POINTWISE_BAR),
                                            "en_rhs": metrics(
                                                np.asarray(tri_args[3])[
                                                    ..., :n_forward],
                                                pre_rhs_n[..., :n_forward],
                                                wet_forward, focus,
                                                POINTWISE_BAR),
                                        }
                                        postlc_l = np.asarray(
                                            solve_kwargs[
                                                "literal_external_rhs"])[
                                                    ..., :nrhs]
                                        postlc_metric = metrics(
                                            postlc_l, postlc_n[
                                                ..., 1:1 + nrhs],
                                            wet_rhs, focus,
                                            POINTWISE_BAR)
                                        postlc_n_int = postlc_n[
                                            ..., 1:1 + nrhs]
                                        budget_associations = {
                                            "left_associated_terms":
                                                postlc_n_int + dt64 * (
                                                    (p_sh2_n + strat_n)
                                                    + diss_rhs_n
                                                ) * wet_rhs.astype(np.float64),
                                            "right_associated_terms":
                                                postlc_n_int + dt64 * (
                                                    p_sh2_n
                                                    + (strat_n + diss_rhs_n)
                                                ) * wet_rhs.astype(np.float64),
                                        }
                                        budget_association_metrics = {
                                            name: metrics(
                                                value,
                                                pre_rhs_n[..., 1:1 + nrhs],
                                                wet_rhs, focus,
                                                POINTWISE_BAR)
                                            for name, value in
                                            budget_associations.items()
                                        }
                                        bottom_ext = (
                                            bottom_level[..., None] + 1)
                                        rhs_bottom_scatter = metrics(
                                            np.take_along_axis(
                                                np.asarray(tri_args[3]),
                                                bottom_ext, axis=-1),
                                            bottom_prod[..., None],
                                            wet2[..., None], focus,
                                            POINTWISE_BAR)
                                        bottom_matrix_identity = {
                                            "a": metrics(
                                                np.take_along_axis(
                                                    a_tri, bottom_ext,
                                                    axis=-1),
                                                np.zeros_like(
                                                    bottom_prod[..., None]),
                                                wet2[..., None], focus,
                                                POINTWISE_BAR),
                                            "diag": metrics(
                                                np.take_along_axis(
                                                    b_tri, bottom_ext,
                                                    axis=-1),
                                                np.ones_like(
                                                    bottom_prod[..., None]),
                                                wet2[..., None], focus,
                                                POINTWISE_BAR),
                                            "c": metrics(
                                                np.take_along_axis(
                                                    c_tri, bottom_ext,
                                                    axis=-1),
                                                np.zeros_like(
                                                    bottom_prod[..., None]),
                                                wet2[..., None], focus,
                                                POINTWISE_BAR),
                                        }
                                        forward_scale = np.sqrt(np.mean(
                                            work_n[..., :n_forward][
                                                wet_forward] ** 2))
                                        forward_bad = wet_forward & (
                                            np.abs(literal_work[
                                                ..., :n_forward]
                                                - work_n[..., :n_forward])
                                            / forward_scale > POINTWISE_BAR)
                                        first_bad = np.argmax(
                                            forward_bad, axis=-1)
                                        bad_columns = np.any(
                                            forward_bad, axis=-1)
                                        offsets = (
                                            first_bad[bad_columns]
                                            - bottom_ext[..., 0][bad_columns])
                                        offset_values, offset_counts = (
                                            np.unique(offsets,
                                                      return_counts=True))
                                        forward_rhs_metric[
                                            "first_bad_level_census"] = {
                                                "nemo_jk_by_python_ext_k": {
                                                    str(k): int(np.count_nonzero(
                                                        bad_columns
                                                        & (first_bad == k)))
                                                    for k in range(n_forward)
                                                    if np.any(bad_columns
                                                              & (first_bad == k))
                                                },
                                                "first_bad_minus_bottom_ext": {
                                                    str(int(k)): int(v)
                                                    for k, v in zip(
                                                        offset_values,
                                                        offset_counts)
                                                },
                                            }
                                        rhs_jk2 = np.asarray(
                                            tri_args[3])[..., 1]
                                        lw_jk2 = a_tri[..., 1]
                                        surface_recip = (
                                            1.0 / np.asarray(surface_n))
                                        nemo_work_jk2 = work_n[..., 1]
                                        jk2_associations = {
                                            "source_divide_then_multiply":
                                                rhs_jk2
                                                - lw_jk2 / surface_recip
                                                * np.float64(1.0),
                                            "factored_lower_times_surface":
                                                rhs_jk2
                                                - lw_jk2
                                                * np.asarray(surface_n),
                                            "multiply_then_divide":
                                                rhs_jk2
                                                - (lw_jk2 * np.float64(1.0))
                                                / surface_recip,
                                        }
                                        jk2_association_metrics = {
                                            name: metrics(
                                                value[..., None],
                                                nemo_work_jk2[..., None],
                                                wet2[..., None], focus,
                                                POINTWISE_BAR)
                                            for name, value in
                                            jk2_associations.items()
                                        }
                                        recurrence_metric = metrics(
                                            literal_recurrence[..., :nrhs],
                                            postsolve_n, wet_rhs, focus,
                                            1.0e-12)
                                        row17m["disposition"] = (
                                            "VERIFIED" if row17m["pass"]
                                            else "DIVERGED")
                                        row17m["exact"] = (
                                            exact_difference_census(
                                                postsolve_l, postsolve_n,
                                                wet_rhs, focus))
                                        row17m["controls"] = planted_controls(
                                            postsolve_n, postsolve_n, wet_rhs,
                                            1.0e-12)
                                        row17m["operand_metrics"] = {
                                            "literal_composite_rhs": rhs_metric,
                                            "matrix": row12_primary,
                                            "rhs_shear": row13m,
                                            "rhs_stratification": row14m,
                                            "rhs_dissipation": row15m,
                                            "posthoc_nemo_literal_recurrence":
                                                recurrence_metric,
                                            "literal_forward_diagonal":
                                                forward_diag_metric,
                                            "literal_forward_rhs":
                                                forward_rhs_metric,
                                            "pre_recurrence_inputs":
                                                pre_stage_metrics,
                                            "post_langmuir_en": postlc_metric,
                                            "budget_from_nemo_post_langmuir":
                                                budget_association_metrics,
                                            "jk2_rhs_association_discriminator":
                                                jk2_association_metrics,
                                            "solver_rhs_bottom_scatter":
                                                rhs_bottom_scatter,
                                            "solver_bottom_matrix_identity":
                                                bottom_matrix_identity,
                                        }
                                        bracket17 = restart_numeric_identity(
                                            args.row17_bracket_instrument_restart,
                                            args.row17_bracket_off_restart)
                                        if not bracket17["pass"]:
                                            raise AssertionError(
                                                "row-17 dump-on/dump-off "
                                                "numeric restart bracket failed")
                                        row17m["instrumentation_bracket"] = bracket17
                                        row17m["instrumentation"] = {
                                            "slot": "tke_dump_en_postsolve.bin",
                                            "location": (
                                                "immediately after "
                                                "zdftke.F90:565 and before "
                                                "nn_etau at :588"),
                                            "sha256": {
                                                str(path): sha256(path)
                                                for path in (
                                                    args.row17_postsolve_dump,
                                                    args.row17_forward_diag_dump,
                                                    args.row17_forward_rhs_dump,
                                                    args.row17_pre_diag_dump,
                                                    args.row17_pre_lower_dump,
                                                    args.row17_pre_rhs_dump,
                                                    args.row17_postlc_dump,
                                                    args.row17_instrument_source,
                                                    args.row17_instrument_binary,
                                                    args.row17_bracket_off_restart,
                                                    args.row17_bracket_instrument_restart,
                                                )
                                            },
                                        }
                                        if not row17m["pass"]:
                                            row4["continuation_preview"][
                                                "first_divergence"] = {
                                                    "row": 17,
                                                    "operation": (
                                                        "TKE tridiagonal "
                                                        "forward/back solve "
                                                        "and floor/mask"),
                                                    "nemo_line": (
                                                        "cfgs/DINO/MY_SRC/"
                                                        "zdftke.F90:547-566"),
                                                    "output": row17m,
                                                    "first_failing_operand": {
                                                        "name": (
                                                            "shared normalized "
                                                            "Thomas recurrence "
                                                            "instead of NEMO's "
                                                            "reciprocal-surface/"
                                                            "jpkm1-terminal "
                                                            "recurrence"),
                                                        "discriminator_pass":
                                                            recurrence_metric[
                                                                "pass"],
                                                        "nemo_line": (
                                                            "cfgs/DINO/MY_SRC/"
                                                            "zdftke.F90:547-565")},
                                                    "operand_localization":
                                                        row17m["operand_metrics"],
                                                    "controls": row17m["controls"],
                                                    "next_round_fix_design": {
                                                        "status": (
                                                            "DESIGNED_NOT_"
                                                            "IMPLEMENTED"),
                                                        "option": (
                                                            "tke_solver_"
                                                            "evaluation"),
                                                        "faithful_default_on_"
                                                        "complete_dino_nemo_"
                                                        "cards": (
                                                            "nemo_literal"),
                                                        "legacy_default_"
                                                        "everywhere_else": (
                                                            "shared_thomas"),
                                                        "legacy_opt_in_on_"
                                                        "dino_nemo_cards": (
                                                            "shared_thomas"),
                                                        "walk_order": [
                                                            "forward diagonal",
                                                            "forward RHS",
                                                            "bottom seed",
                                                            "back substitution",
                                                            "rn_emin floor",
                                                            "wmask",
                                                        ],
                                                        "rule": (
                                                            "Transcribe the "
                                                            "reciprocal surface "
                                                            "seed, two forward "
                                                            "recurrences, direct "
                                                            "jpkm1 seed, reverse "
                                                            "substitution and "
                                                            "postsolve floor/mask "
                                                            "as JAX scans; keep "
                                                            "the shared solver "
                                                            "untouched elsewhere."),
                                                        "reason_for_stop": (
                                                            "A new differentiable "
                                                            "solver path plus "
                                                            "JIT/grad and boundary "
                                                            "tests is a single fix "
                                                            "too large to add after "
                                                            "localization in this "
                                                            "round."),
                                                    },
                                                }
                                    row4["continuation_preview"]["rows"][
                                        "17_tke_tridiagonal_solve"] = row17m
                                    if (row17m.get("pass")
                                            and tke_capture["etau_calls"]):
                                        etau_args, etau_kwargs, etau_out = (
                                            tke_capture["etau_calls"][-1])
                                        if len(etau_args) != 4:
                                            raise AssertionError(
                                                "unexpected etau call contract")
                                        pre_etau_l = np.asarray(etau_args[0])
                                        post_etau_l = np.asarray(etau_out)
                                        post_etau_n_full = base._load_interior(
                                            str(RUN / "tke_dump_en.bin"),
                                            ni, nj)
                                        netau = min(
                                            post_etau_l.shape[-1],
                                            post_etau_n_full.shape[-1] - 1,
                                            wet_w_all.shape[-1])
                                        post_etau_n = post_etau_n_full[
                                            ..., 1:1 + netau]
                                        wet_etau = wet_w_all[..., :netau]
                                        row18m = metrics(
                                            post_etau_l[..., :netau],
                                            post_etau_n, wet_etau, focus,
                                            POINTWISE_BAR)
                                        pre_etau_n = postsolve_n[..., :netau]
                                        increment_l = (
                                            post_etau_l[..., :netau]
                                            - pre_etau_l[..., :netau])
                                        increment_n = (
                                            post_etau_n - pre_etau_n)
                                        raw_lat_rad = jnp.asarray(
                                            model.grid.lat_T,
                                            dtype=etau_args[0].dtype)
                                        raw_htau = jnp.maximum(
                                            tke_mod._NEMO_TKE_HTAU_MIN_M,
                                            jnp.minimum(
                                                tke_mod._NEMO_TKE_HTAU_MAX_M,
                                                tke_mod._NEMO_TKE_HTAU_SLOPE_M
                                                * jnp.abs(jnp.sin(
                                                    raw_lat_rad))))
                                        etau_rho0 = etau_kwargs["rho_0"]
                                        surface_for_etau = jnp.maximum(
                                            tke_mod._NEMO_TKE_EMIN0,
                                            tke_mod._NEMO_TKE_EBB / etau_rho0
                                            * jnp.maximum(etau_args[1], 0.0))
                                        raw_rad_increment = (
                                            (etau_args[3].etau_frac
                                             * surface_for_etau[..., None])
                                            * jnp.exp(
                                                -etau_args[2]
                                                / raw_htau[..., None]))
                                        etau_ice = etau_kwargs["ice_frac"]
                                        if etau_ice is not None:
                                            raw_rad_increment = (
                                                raw_rad_increment
                                                * jnp.maximum(
                                                    0.0,
                                                    1.0
                                                    - etau_ice[..., None]))
                                        raw_rad_candidate = np.asarray(
                                            etau_args[0]
                                            + raw_rad_increment)
                                        raw_rad_metric = metrics(
                                            raw_rad_candidate[..., :netau],
                                            post_etau_n, wet_etau, focus,
                                            POINTWISE_BAR)
                                        row18_dump_args = (
                                            args.row18_argument_dump,
                                            args.row18_exp_dump,
                                            args.row18_increment_dump,
                                            args.row18_bracket_instrument_restart,
                                            args.row18_instrument_source,
                                            args.row18_instrument_binary,
                                        )
                                        if any(x is None for x in
                                               row18_dump_args):
                                            raise AssertionError(
                                                "row 18 requires the direct "
                                                "argument/exp/increment dumps and "
                                                "instrument provenance")
                                        for path in row18_dump_args:
                                            if not path.is_file():
                                                raise FileNotFoundError(path)
                                        arg_n_full = base._load_interior(
                                            str(args.row18_argument_dump), ni, nj)
                                        exp_n_full = base._load_interior(
                                            str(args.row18_exp_dump), ni, nj)
                                        inc_direct_n_full = (
                                            base._load_interior(
                                                str(args.row18_increment_dump),
                                                ni, nj))
                                        degree_htau = jnp.maximum(
                                            tke_mod._NEMO_TKE_HTAU_MIN_M,
                                            jnp.minimum(
                                                tke_mod._NEMO_TKE_HTAU_MAX_M,
                                                tke_mod._NEMO_TKE_HTAU_SLOPE_M
                                                * jnp.abs(jnp.sin(jnp.deg2rad(
                                                    etau_kwargs[
                                                        "lat_deg"])))))
                                        argument_plus_exp_l = np.asarray(jnp.exp(
                                            -etau_args[2]
                                            / degree_htau[..., None]))[
                                                ..., :netau]
                                        argument_l = np.asarray(
                                            -etau_args[2]
                                            / degree_htau[..., None])[
                                                ..., :netau]
                                        argument_n = arg_n_full[
                                            ..., 1:1 + netau]
                                        exp_n = exp_n_full[..., 1:1 + netau]
                                        isolated_exp_l = np.asarray(jnp.exp(
                                            jnp.asarray(argument_n)))
                                        literal_dump_exp = np.asarray(
                                            tke_mod._nemo_glibc234_vector_exp(
                                                jnp.asarray(argument_n)))
                                        literal_production_exp = np.asarray(
                                            tke_mod._nemo_glibc234_vector_exp(
                                                jnp.asarray(argument_l)))
                                        inc_direct_n = inc_direct_n_full[
                                            ..., 1:1 + netau]
                                        argument_metric = metrics(
                                            argument_l, argument_n, wet_etau,
                                            focus, POINTWISE_BAR)
                                        exp_metric = metrics(
                                            isolated_exp_l, exp_n, wet_etau,
                                            focus, POINTWISE_BAR)
                                        literal_dump_exp_metric = metrics(
                                            literal_dump_exp, exp_n, wet_etau,
                                            focus, POINTWISE_BAR)
                                        argument_bits = np.asarray(
                                            argument_n).view(np.uint64)
                                        ordinary_exp = wet_etau & (
                                            ((argument_bits
                                              & np.uint64(
                                                  0x7FFF_FFFF_FFFF_FFFF))
                                             >> np.uint64(32))
                                            <= np.uint64(0x4086_232A))
                                        literal_bits = np.asarray(
                                            literal_dump_exp).view(np.uint64)
                                        nemo_exp_bits = np.asarray(
                                            exp_n).view(np.uint64)
                                        literal_exp_exact = {
                                            "ordinary_wet_elements": int(
                                                ordinary_exp.sum()),
                                            "ordinary_bit_mismatches": int(
                                                np.sum(ordinary_exp & (
                                                    literal_bits
                                                    != nemo_exp_bits))),
                                            "exceptional_wet_elements": int(
                                                np.sum(wet_etau
                                                       & ~ordinary_exp)),
                                            "exceptional_bit_mismatches": int(
                                                np.sum(wet_etau
                                                       & ~ordinary_exp
                                                       & (literal_bits
                                                          != nemo_exp_bits))),
                                            "exceptional_max_abs": float(
                                                np.max(np.where(
                                                    wet_etau & ~ordinary_exp,
                                                    np.abs(literal_dump_exp
                                                           - exp_n), 0.0))),
                                            "range_guard_high_word": (
                                                "0x4086232a"),
                                        }
                                        literal_production_exp_metric = metrics(
                                            literal_production_exp, exp_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        argument_plus_exp_metric = metrics(
                                            argument_plus_exp_l, exp_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        nemo_radian = (
                                            jnp.asarray(
                                                float.fromhex(
                                                    "0x1.1df46a2529d39p-6"),
                                                dtype=etau_args[0].dtype)
                                            * etau_kwargs["lat_deg"])
                                        literal_degree_htau = jnp.maximum(
                                            tke_mod._NEMO_TKE_HTAU_MIN_M,
                                            jnp.minimum(
                                                tke_mod._NEMO_TKE_HTAU_MAX_M,
                                                tke_mod._NEMO_TKE_HTAU_SLOPE_M
                                                * jnp.abs(jnp.sin(
                                                    nemo_radian))))
                                        literal_degree_argument = np.asarray(
                                            -etau_args[2]
                                            / literal_degree_htau[..., None])[
                                                ..., :netau]
                                        literal_degree_argument_metric = metrics(
                                            literal_degree_argument,
                                            argument_n, wet_etau, focus,
                                            POINTWISE_BAR)
                                        raw_radian_argument = np.asarray(
                                            -etau_args[2]
                                            / raw_htau[..., None])[..., :netau]
                                        raw_radian_argument_metric = metrics(
                                            raw_radian_argument, argument_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        rounded = tke_mod._nemo_binary64_round
                                        literal_line_increment = rounded(
                                            etau_args[3].etau_frac
                                            * surface_for_etau[..., None])
                                        literal_line_increment = rounded(
                                            literal_line_increment
                                            * jnp.asarray(exp_n))
                                        if etau_ice is not None:
                                            literal_line_increment = rounded(
                                                literal_line_increment
                                                * jnp.maximum(
                                                    0.0, 1.0
                                                    - etau_ice[..., None]))
                                        literal_line_increment = np.asarray(
                                            literal_line_increment)[
                                                ..., :netau]
                                        literal_line_candidate = np.asarray(
                                            rounded(
                                                jnp.asarray(pre_etau_l[
                                                    ..., :netau])
                                                + jnp.asarray(
                                                    literal_line_increment)))
                                        literal_line_increment_metric = metrics(
                                            literal_line_increment,
                                            inc_direct_n, wet_etau, focus,
                                            POINTWISE_BAR)
                                        literal_line_candidate_metric = metrics(
                                            literal_line_candidate,
                                            post_etau_n, wet_etau, focus,
                                            POINTWISE_BAR)
                                        numpy_exp_degree = np.exp(
                                            -np.asarray(etau_args[2])
                                            / np.asarray(degree_htau)[
                                                ..., None])[..., :netau]
                                        numpy_exp_raw_rad = np.exp(
                                            -np.asarray(etau_args[2])
                                            / np.asarray(raw_htau)[
                                                ..., None])[..., :netau]
                                        numpy_exp_degree_metric = metrics(
                                            numpy_exp_degree, exp_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        numpy_exp_raw_rad_metric = metrics(
                                            numpy_exp_raw_rad, exp_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        numpy_candidate_increment = (
                                            np.asarray(
                                                etau_args[3].etau_frac
                                                * surface_for_etau)[..., None]
                                            * numpy_exp_degree)
                                        if etau_ice is not None:
                                            numpy_candidate_increment *= (
                                                np.maximum(
                                                    0.0, 1.0 - np.asarray(
                                                        etau_ice)[..., None]))
                                        numpy_candidate = (
                                            pre_etau_l[..., :netau]
                                            + numpy_candidate_increment)
                                        numpy_candidate_metric = metrics(
                                            numpy_candidate, post_etau_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        inc_direct_metric = metrics(
                                            increment_l, inc_direct_n,
                                            wet_etau, focus, POINTWISE_BAR)
                                        bracket18 = restart_numeric_identity(
                                            args.row18_bracket_instrument_restart,
                                            args.row17_bracket_off_restart)
                                        if not bracket18["pass"]:
                                            raise AssertionError(
                                                "row-18 dump-on/dump-off "
                                                "numeric restart bracket failed")
                                        with xr.open_dataset(
                                                RUN / "mesh_mask.nc",
                                                decode_cf=False) as mesh_ds:
                                            gphit_n = np.asarray(
                                                mesh_ds["gphit"]).squeeze()
                                        lat_deg_l = np.asarray(
                                            etau_kwargs["lat_deg"])
                                        lat_operand_metric = metrics(
                                            lat_deg_l[..., None],
                                            gphit_n[..., None],
                                            wet2[..., None], focus,
                                            POINTWISE_BAR)
                                        row18m["disposition"] = (
                                            "VERIFIED" if row18m["pass"]
                                            else "DIVERGED")
                                        row18m["exact"] = (
                                            exact_difference_census(
                                                post_etau_l[..., :netau],
                                                post_etau_n, wet_etau, focus))
                                        row18m["controls"] = planted_controls(
                                            post_etau_n, post_etau_n,
                                            wet_etau, POINTWISE_BAR)
                                        row18m["operand_metrics"] = {
                                            "pre_etau_en": metrics(
                                                pre_etau_l[..., :netau],
                                                pre_etau_n, wet_etau, focus,
                                                1.0e-12),
                                            "etau_increment": metrics(
                                                increment_l, increment_n,
                                                wet_etau, focus,
                                                POINTWISE_BAR),
                                            "direct_nemo_exponent_argument":
                                                argument_metric,
                                            "direct_nemo_exp": exp_metric,
                                            "literal_exp_on_dumped_argument":
                                                literal_dump_exp_metric,
                                            "literal_exp_bitwise_census":
                                                literal_exp_exact,
                                            "literal_exp_on_production_argument":
                                                literal_production_exp_metric,
                                            "production_argument_plus_exp":
                                                argument_plus_exp_metric,
                                            "literal_rpi_argument":
                                                literal_degree_argument_metric,
                                            "raw_radian_argument":
                                                raw_radian_argument_metric,
                                            "literal_line_dumped_exp_increment":
                                                literal_line_increment_metric,
                                            "literal_line_dumped_exp_output":
                                                literal_line_candidate_metric,
                                            "diagnostic_numpy_libm_exp_degree":
                                                numpy_exp_degree_metric,
                                            "diagnostic_numpy_libm_exp_raw_rad":
                                                numpy_exp_raw_rad_metric,
                                            "substitute_numpy_libm_exp":
                                                numpy_candidate_metric,
                                            "direct_nemo_increment":
                                                inc_direct_metric,
                                            "gphit_after_degree_roundtrip":
                                                lat_operand_metric,
                                            "substitute_raw_tgrid_radians":
                                                raw_rad_metric,
                                            "selector": {
                                                "nn_etau": 1,
                                                "etau_mode":
                                                    etau_args[3].etau_mode,
                                                "htau_mode": etau_args[
                                                    3].etau_htau_mode,
                                                "nemo_line": (
                                                    "cfgs/DINO/MY_SRC/"
                                                    "zdftke.F90:588-592"),
                                            },
                                        }
                                        row18m["instrumentation_bracket"] = (
                                            bracket18)
                                        row18m["instrumentation"] = {
                                            "slots": [
                                                "tke_dump_etau_argument.bin",
                                                "tke_dump_etau_exp.bin",
                                                "tke_dump_etau_increment.bin",
                                            ],
                                            "location": (
                                                "active nn_etau=1 expression "
                                                "at zdftke.F90:590-591"),
                                            "sha256": {
                                                str(path): sha256(path)
                                                for path in row18_dump_args
                                            },
                                        }
                                        row4["continuation_preview"]["rows"][
                                            "18_etau_penetration"] = row18m
                                        if not row18m["pass"]:
                                            row4["continuation_preview"][
                                                "first_divergence"] = {
                                                    "row": 18,
                                                    "operation": (
                                                        "nn_etau=1 surface-TKE "
                                                        "penetration"),
                                                    "nemo_line": (
                                                        "cfgs/DINO/MY_SRC/"
                                                        "zdftke.F90:588-592"),
                                                    "output": row18m,
                                                    "first_failing_operand": {
                                                        "name": (
                                                            "production "
                                                            "-gdepw/htau input; "
                                                            "gdepw versus htau "
                                                            "not yet separated"),
                                                        "nemo_line": (
                                                            "cfgs/DINO/MY_SRC/"
                                                            "zdftke.F90:590"),
                                                    },
                                                    "operand_localization":
                                                        row18m[
                                                            "operand_metrics"],
                                                    "controls": row18m[
                                                        "controls"],
                                                    "next_round_fix_design": {
                                                        "status": "DESIGNED",
                                                        "option": (
                                                            "instrument direct "
                                                            "gdepw and htau "
                                                            "operands before "
                                                            "choosing the next "
                                                            "production option"),
                                                        "faithful_default_on_"
                                                        "complete_dino_nemo_"
                                                        "cards": (
                                                            "nemo_literal"),
                                                        "legacy_default_"
                                                        "everywhere_else": (
                                                            "jax_expression"),
                                                        "rule": (
                                                            "Add write-only htau "
                                                            "and live-gdepw dumps, "
                                                            "prove restart bit "
                                                            "identity, substitute "
                                                            "one operand at a time, "
                                                            "then implement only "
                                                            "the first failing "
                                                            "construction."),
                                                    },
                                                }
                                            ordered_tail = {
                                                19: "buoyancy mixing length",
                                                20: "nn_mxl=3 length limiting",
                                                21: "base avm/avt/dissl assembly",
                                                22: "inverse-Prandtl avt correction",
                                                23: "closure coefficient copy",
                                                24: "river-mouth enhancement",
                                                25: "EVD tracer overwrite",
                                                26: "EVD momentum overwrite",
                                                27: "avs and wave enhancements",
                                                28: "composed-avt turbocline scan",
                                                29: "avm lateral-boundary update",
                                                30: "ldf_slp",
                                                31: "momentum implicit solve",
                                                32: "tracer implicit solve",
                                            }
                                            for number, operation in (
                                                    ordered_tail.items()):
                                                row4[
                                                    "continuation_preview"
                                                ]["rows"][
                                                    f"{number}_{operation.replace(' ', '_')}"
                                                ] = {
                                                    "disposition": "UNMEASURED",
                                                    "reason": (
                                                        "ordered stop at row 18"
                                                    ),
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
        HERE / "PREREG_zdf_chain_sweep_round6.md",
        HERE / "PREREG_zdf_chain_sweep_round7.md",
        HERE / "PREREG_zdf_chain_sweep_round8.md",
        HERE / "PREREG_zdf_chain_sweep_round9.md",
        HERE / "PREREG_zdf_chain_sweep_round10.md",
        HERE / "PREREG_zdf_chain_sweep_round10_update.md",
        HERE / "PREREG_zdf_chain_sweep_round11.md",
        HERE / "PREREG_zdf_chain_sweep_round12.md",
        HERE / "PREREG_zdf_chain_sweep_round13.md",
        HERE / "PREREG_zdf_chain_sweep_round14.md",
        HERE / "PREREG_zdf_chain_sweep_round14_update.md",
        HERE / "zdf_etau_libm_ownership.py",
        HERE / "kamm_twin_90d.py",
        Path("packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
        Path("packages/ocean/legoesm/ocean/eos.py"),
        Path("packages/ocean/legoesm/ocean/experiments/dino.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/nemo_io.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py"),
        Path("packages/ocean/legoesm/ocean/fidelity/time_levels.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/"
             "_glibc234_exp_table.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py"),
        Path("packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py"),
        Path("/usr/lib64/libmvec.so.1"),
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
        RUN / "tke_dump_zri.bin",
        RUN / "tke_dump_en.bin",
        RUN / "dump_nmln.bin",
        RUN / "dump_hmlp.bin",
        RUN / "drg_dump_rCdU_bot.bin",
        RUN / "sbc_dump_utau.bin",
        RUN / "eiv_dump_gdept.bin", RUN / "eiv_dump_e3w.bin", args.mld_maps,
    ]
    if args.bn2_chain_dir is not None:
        input_paths.extend([
            *(args.bn2_chain_dir / f"bn2_dump_{name}.bin" for name in
              ("zrw", "zaw", "zbw", "numerator", "result")),
            args.bn2_bracket_off_restart,
            args.bn2_bracket_instrument_restart,
            args.bn2_certified_restart,
        ])
    round9_instrument_paths = tuple(path for path in (
        args.row17_postsolve_dump,
        args.row17_forward_diag_dump,
        args.row17_forward_rhs_dump,
        args.row17_pre_diag_dump,
        args.row17_pre_lower_dump,
        args.row17_pre_rhs_dump,
        args.row17_postlc_dump,
        args.row17_bracket_off_restart,
        args.row17_bracket_instrument_restart,
        args.row17_instrument_source,
        args.row17_instrument_binary,
        args.row18_argument_dump,
        args.row18_exp_dump,
        args.row18_increment_dump,
        args.row18_bracket_instrument_restart,
        args.row18_instrument_source,
        args.row18_instrument_binary,
    ) if path is not None)
    input_paths.extend(round9_instrument_paths)
    continuation = row4.get("continuation_preview")
    rows = {"1_eos_rab_before": row1, "2_bn2_before": row2,
            "3_eos_rab_bn2_now": row3, "4_zdf_sh2": row4}
    if continuation is not None:
        for name, value in continuation["rows"].items():
            disposition = value.get("disposition")
            if disposition is None:
                disposition = "VERIFIED" if value["pass"] else "DIVERGED"
            rows[name] = {
                "output": value,
                "disposition": disposition,
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
        "schema": "zdf-chain-sweep-v9",
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
                         "tke_dump_dissl.bin",
                         "tke_dump_zri.bin", "tke_dump_pdlr.bin",
                         "tke_dump_en.bin",
                         "tke_dump_en_postlc.bin",
                         "tke_dump_zdiag_pre.bin",
                         "tke_dump_zlw_pre.bin",
                         "tke_dump_en_pre.bin",
                         "tke_dump_zdiag_forward.bin",
                         "tke_dump_zrhs_forward.bin",
                         "tke_dump_en_postsolve.bin",
                         "tke_dump_etau_argument.bin",
                         "tke_dump_etau_exp.bin",
                         "tke_dump_etau_increment.bin",
                         "dump_nmln.bin", "dump_hmlp.bin",
                         "drg_dump_rCdU_bot.bin", "sbc_dump_utau.bin",
                         "eiv_dump_gdept.bin", "eiv_dump_e3w.bin",
                         "bn2_dump_zrw.bin", "bn2_dump_zaw.bin",
                         "bn2_dump_zbw.bin", "bn2_dump_numerator.bin",
                         "bn2_dump_result.bin")},
        "focus_columns_ji": [list(x) for x in focus],
        "bars": {"pointwise_column": POINTWISE_BAR, "corr": CORR_BAR,
                 "rms_ratio_epsilon": RATIO_EPS},
        "controls": controls,
        "rows": rows,
        "first_divergence": first_divergence,
        "prior_committed_receipts": {
            "provenance_commit": "7de1be2cc74",
            "artifact_sha256": (
                "8b001296c8fc8c0fa72ba202a52bc43a89b3e775bf39ade3644da196545444ec"),
            "ordered_status": (
                "post-hoc receipts only; rows 11+ are not promoted past "
                "the corrected row-10 stop"),
            "row11": {
                "composite": "0/9920 at 1e-15; max 2.877796e-16",
                "tracked_qualification": (
                    "intermediate bn2 numerator misses 3/9920; max "
                    "1.174266e-15"),
            },
            "row12": {
                "matrix": "0/9920 at 1e-15; max 0",
                "coefficients": "zd_up=0/9920; zd_lw=0/9920; zdiag=0/9920",
                "line_510_dissl_operand": "0/9920; max 0",
            },
        },
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
                disposition = cm.get("disposition")
                if disposition is None:
                    disposition = "VERIFIED" if cm["pass"] else "DIVERGED"
                if "max_column_error" in cm:
                    print(f"row {name}: {disposition} "
                          f"max={cm['max_column_error']:.6e} "
                          f"bad_columns={cm['n_diverged_columns']}/"
                          f"{cm['n_wet_columns']}")
                else:
                    print(f"row {name}: {disposition} reason={cm['reason']}")
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
                for name, cm in first["output"].get(
                        "single_operand_substitutions", {}).items():
                    print(f"    substitute {name}: pass={cm['pass']} "
                          f"max={cm['max_column_error']:.6e} "
                          f"bad_columns={cm['n_diverged_columns']}/"
                          f"{cm['n_wet_columns']}")
                if "first_passing_substitution" in first["output"]:
                    print("    first_passing_substitution="
                          f"{first['output']['first_passing_substitution']}")
                chain = first["output"].get("bn2_operand_chain")
                if chain is not None:
                    print("    bn2 first bitwise divergence="
                          f"{chain['first_bitwise_diverging_checkpoint']}")
                    for name, cm in chain["ordered_checkpoints"].items():
                        exact = cm["exact"]
                        print(f"    bn2 {name}: pass={cm['pass']} "
                              f"bad_columns={cm['n_diverged_columns']}/"
                              f"{cm['n_wet_columns']} exact_bad_columns="
                              f"{exact['n_unequal_wet_columns']}")
                    print("    first zri-closing bn2 substitution="
                          f"{chain['first_zri_closing_substitution']}")
                    print("    dump-on/off bracket pass="
                          f"{chain['instrumentation_bracket']['pass']}")
                    print("    certified all-numeric claim retracted="
                          f"{chain['certified_all_numeric_identity_claim_retracted']}")
    if localization:
        for c in localization["candidates_in_nemo_evaluation_order"]:
            print(f"  substitute {c['substitution']}: pass={c['metrics']['pass']} "
                  f"max={c['metrics']['max_column_error']:.6e}")
        print(f"  first_passing_substitution={localization['first_passing_substitution']}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
