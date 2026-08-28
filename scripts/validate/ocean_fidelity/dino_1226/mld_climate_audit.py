#!/usr/bin/env python
"""Offline climate-level DINO mixed-layer-depth audit against NEMO 5.0.2.

This probe applies the already-certified production transcription of NEMO's
``zdf_mxl`` density-integral criterion symmetrically to matching NOW-level
T/S/ssh states. This is a standardized offline state diagnostic, not NEMO's
online time-level combination: executed DINO builds ``rn2b`` from BEFORE T/S
and combines it with NOW geometry. Because the legoESM snapshots do not carry
BEFORE T/S, the native online combination cannot be scored symmetrically. An
unscored NEMO BEFORE-vs-NOW sensitivity is emitted to bound that limitation.
The probe does not reimplement the MLD criterion and does not substitute a
density diagnostic for the ``avt``-based turbocline ``hmld``.

Oracle source (NEMO 5.0.2, executed DINO configuration):

* ``cfgs/DINO/WORK/zdfmxl.F90:90-104``: ``hmlp`` integrates
  ``MAX(rn2b,0)*e3w`` from ``nlb10`` against ``grav*rho_c/rho0`` and returns
  live ``gdepw(nmln)``; ``rho_c=0.01`` is at line 34.
* ``cfgs/DINO/WORK/domzgr.F90:367-371``: nominal 10 m reference and ``nlb10``.
* ``cfgs/DINO/WORK/zdfmxl.F90:123-152``: turbocline ``hmld`` scans composed
  ``avt`` against ``avt_c=5e-4`` (line 35). The supplied snapshots lack
  ``avt``, so that diagnostic remains UNMEASURED.
* ``cfgs/DINO/WORK/stpmlf.F90:204-210``: online ``rn2b`` comes from BEFORE
  T/S while ``zdf_phy`` receives the NOW geometry level.

Run from the repository root, on CPU only::

  PYTHONPATH="$(find packages -mindepth 1 -maxdepth 1 -type d | paste -sd:):src" \
    python scripts/validate/ocean_fidelity/dino_1226/mld_climate_audit.py \
      --git-dir /tmp/codex-mld-audit-git2 \
      --out-dir /tmp/dino_mld_audit_codex
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import eos as eos_mod
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _nemo_mld_from_n2_integral,
    _nemo_native_active_3d,
)
from legoesm.ocean.vertical import compute_ocean_jacobian

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG_mld_climate_audit.md"
NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
RUN = NEMO_ROOT / "cfgs/DINO/RUN_VERDICT360_M0"
MESH = NEMO_ROOT / "cfgs/DINO/RUN_TRAJ/mesh_mask.nc"
WORK = NEMO_ROOT / "cfgs/DINO/WORK"
MY_SRC = NEMO_ROOT / "cfgs/DINO/MY_SRC"
DAYS = (0, 30, 60, 90)
KT0 = 5760
STEPS_PER_DAY = 32
H1 = 10.13875112538517
ARMS = {
    "basin_legacy": Path(
        "/tmp/codex-basin-rect/results/dino_1455/tcarry_basin90_legacy.npz"),
    "basin_stagger_corrected": Path(
        "/tmp/codex-basin-rect/results/dino_1455/tcarry_basin90_corrected.npz"),
    "euc_tke_prefix": Path(
        "/tmp/dino_euc_mechanism/twin90_prefix/twin90_prefix.npz"),
    "euc_tke_floor_fix": Path(
        "/tmp/dino_euc_mechanism/twin90_fix/twin90_fix.npz"),
}
LOGS = {
    "basin_legacy": ARMS["basin_legacy"].with_suffix(".log"),
    "basin_stagger_corrected": ARMS["basin_stagger_corrected"].with_suffix(".log"),
    "euc_tke_prefix": Path("/tmp/dino_euc_mechanism/twin90_prefix.log"),
    "euc_tke_floor_fix": Path("/tmp/dino_euc_mechanism/twin90_fix.log"),
}
REGION_ROWS = {
    "channel": np.arange(14, 49),
    "basin": np.arange(0, 14),
    "equator": np.asarray([99]),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(git_dir: str | None, *args: str) -> str:
    cmd = ["git"]
    if git_dir:
        cmd.append(f"--git-dir={git_dir}")
    cmd.extend(args)
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def git_stamp(git_dir: str | None) -> dict[str, object]:
    head = git_output(git_dir, "rev-parse", "HEAD")
    status = git_output(git_dir, "status", "--porcelain", "--untracked-files=no")
    require(not status, f"tracked audit worktree is dirty:\n{status}")
    return {"head": head, "dirty_tracked_files": 0}


def nemo_git_stamp() -> dict[str, object]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(NEMO_ROOT), "rev-parse", "HEAD"], text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(NEMO_ROOT), "status", "--porcelain",
             "--untracked-files=no"], text=True,
        ).splitlines()
        return {"head": head, "dirty_tracked_files": len(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"head": "UNAVAILABLE", "dirty_tracked_files": "UNAVAILABLE"}


def restart_paths(day: int) -> list[Path]:
    kt = KT0 + STEPS_PER_DAY * day
    paths = [Path(p) for p in sorted(glob.glob(str(RUN / f"DINO_{kt:08d}_restart*.nc")))]
    require(paths, f"no NEMO restart files for day {day}, kt={kt}")
    return paths


def load_script_module(filename: str, module_name: str):
    path = ROOT / "scripts/validate/ocean_fidelity" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REBUILD = load_script_module(
    "rebuild_nemo_restart.py", "_mld_restart_rebuild").rebuild
VERTICAL_LADDER_SHA256 = load_script_module(
    "dino_1226/kamm_twin_90d.py", "_mld_kamm_twin").vertical_ladder_sha256


def scalar(d: np.lib.npyio.NpzFile, key: str):
    require(key in d.files, f"missing required NPZ stamp {key}")
    a = d[key]
    require(a.shape == (), f"stamp {key} must be scalar, got {a.shape}")
    return a.item()


def validate_arm(name: str, path: Path, log_path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing arm {path}")
    require(log_path.is_file(), f"missing producer log {log_path}")
    with np.load(path, allow_pickle=False) as d:
        for day in DAYS:
            for field in ("T3d", "S3d", "eta3d"):
                require(f"{field}_day{day}" in d.files,
                        f"{name}: missing {field}_day{day}")
        require(bool(scalar(d, "stable")), f"{name}: producer marked unstable")
        require(int(scalar(d, "blew_up_at_step")) == -1,
                f"{name}: producer blew up")
        require(str(scalar(d, "control_dtype")) == "float64",
                f"{name}: non-fp64 control")
        require(str(scalar(d, "nemo_ladder_mode")) == "both",
                f"{name}: non-certified ladder")
        require(str(scalar(d, "twin_start_mode")) == "bridged",
                f"{name}: non-bridged start")
        require(float(scalar(d, "seasonal_t0_seconds")) ==
                float(scalar(d, "seasonal_t0_reference_seconds")),
                f"{name}: legacy/out-of-season clock")
        if "producer_dirty_tracked_files" in d.files:
            require(int(scalar(d, "producer_dirty_tracked_files")) == 0,
                    f"{name}: dirty producer recorded in NPZ")
        ladder_sha = str(scalar(d, "vertical_ladder_sha256"))
        require(re.fullmatch(r"[0-9a-f]{64}", ladder_sha) is not None,
                f"{name}: malformed vertical ladder SHA-256")
        run_config = json.loads(str(scalar(d, "run_config")))
        land_mask = np.asarray(d["land_mask"], dtype=np.float64)

    first = log_path.read_text(errors="replace").splitlines()[0]
    match = re.fullmatch(
        r"PROVENANCE: HEAD=([0-9a-f]{40}) dirty_tracked_files=([0-9]+)", first)
    require(match is not None, f"{name}: malformed producer provenance line")
    require(int(match.group(2)) == 0, f"{name}: dirty producer recorded in log")
    return {
        "producer_git_sha": match.group(1),
        "producer_dirty_tracked_files": int(match.group(2)),
        "vertical_ladder_sha256": ladder_sha,
        "run_config": run_config,
        "land_mask": land_mask,
    }


def synthetic_controls(g: float, rho0: float, rho_c: float) -> dict[str, object]:
    """Hand-computed crossing and planted violation through production code."""
    nlev = 6
    dz = np.full(nlev, 10.0, dtype=np.float64)
    gdept = np.cumsum(dz) - 0.5 * dz
    z = types.SimpleNamespace(
        dz_ref=jnp.asarray(dz),
        t_depth_ref=jnp.asarray(gdept),
        is_active=jnp.asarray(np.ones((2, 2, nlev), dtype=np.float64)),
        h_partial=jnp.asarray(np.broadcast_to(dz, (2, 2, nlev))),
    )
    active = np.ones((2, 2, nlev), dtype=np.float64)
    active[..., -1] = 0.0
    temp = jnp.zeros((2, 2, nlev), dtype=jnp.float64)
    salt = jnp.zeros_like(temp)
    mask = jnp.ones((2, 2), dtype=jnp.float64)
    threshold = g * rho_c / rho0

    def run(first_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        contributions = np.zeros(nlev - 1, dtype=np.float64)
        contributions[0] = first_fraction * threshold
        contributions[1] = 0.02 * threshold
        # Baseline first=0.99 stays below at level 0, then crosses at level 1.
        # Violation first=1.01 crosses immediately at level 0.
        n2 = np.broadcast_to(contributions / np.diff(gdept), (2, 2, nlev - 1))

        def prescribed_bn2(*_args, **_kwargs):
            return jnp.asarray(n2)

        original = eos_mod.compute_buoyancy_frequency_nemo_bn2
        eos_mod.compute_buoyancy_frequency_nemo_bn2 = prescribed_bn2
        try:
            hml, base = _nemo_mld_from_n2_integral(
                temp, salt, mask, z, None, rho_c, g, rho0,
                active_3d=jnp.asarray(active), jacobian=None,
            )
        finally:
            eos_mod.compute_buoyancy_frequency_nemo_bn2 = original
        return np.asarray(hml), np.asarray(base)

    baseline_hml, baseline_base = run(0.99)
    planted_hml, planted_base = run(1.01)
    require(np.all(baseline_base == 1),
            f"synthetic hand level failed: {baseline_base}")
    require(np.all(baseline_hml == 20.0),
            f"synthetic hand depth failed: {baseline_hml}")
    require(np.all(planted_base == 0) and np.all(planted_hml == 10.0),
            "planted threshold violation did not change the MLD level")
    false_expectation_fired = False
    try:
        require(np.all(planted_base == 1),
                "deliberately false expected planted level")
    except RuntimeError:
        false_expectation_fired = True
    require(false_expectation_fired,
            "deliberately false expected level did not fire")
    return {
        "threshold_m_s2": threshold,
        "baseline_fraction": 0.99,
        "baseline_base_index": 1,
        "baseline_mld_m": 20.0,
        "planted_fraction": 1.01,
        "planted_base_index": 0,
        "planted_mld_m": 10.0,
        "false_expected_level_fired": bool(false_expectation_fired),
    }


def compute_mld(temp, salt, eta, mask, h_bathy, z_coord, eos_fn, mc, active_3d):
    temp_j = jnp.asarray(temp, dtype=jnp.float64)
    salt_j = jnp.asarray(salt, dtype=jnp.float64)
    etaj = jnp.asarray(eta, dtype=jnp.float64)
    jac = compute_ocean_jacobian(etaj, h_bathy, z_coord)
    hml, base = _nemo_mld_from_n2_integral(
        temp_j, salt_j, mask, z_coord, eos_fn, mc.gm_redi.mld_rho_c,
        mc.g, mc.rho_0, active_3d=active_3d, jacobian=jac,
    )
    return np.asarray(hml, dtype=np.float64), np.asarray(base, dtype=np.int32)


def region_metrics(lego: np.ndarray, nemo: np.ndarray, weights: np.ndarray,
                   common_wet: np.ndarray) -> dict[str, dict[str, float]]:
    require(lego.dtype == np.float64 and nemo.dtype == np.float64,
            "MLD maps must be float64")
    result: dict[str, dict[str, float]] = {}
    for name, rows in REGION_ROWS.items():
        select = np.zeros(common_wet.shape, dtype=bool)
        select[rows, :] = True
        wet = select & common_wet
        require(bool(wet.any()), f"empty region {name}")
        require(np.isfinite(lego[wet]).all() and np.isfinite(nemo[wet]).all(),
                f"non-finite wet MLD in {name}")
        require(np.isfinite(weights[wet]).all() and np.all(weights[wet] > 0.0),
                f"invalid area weights in {name}")
        w = weights[wet]
        sw = float(w.sum())
        diff = lego[wet] - nemo[wet]
        result[name] = {
            "wet_cells": int(wet.sum()),
            "weight_sum_m2": sw,
            "legoesm_mean_m": float(np.sum(w * lego[wet]) / sw),
            "nemo_mean_m": float(np.sum(w * nemo[wet]) / sw),
            "mean_bias_m": float(np.sum(w * diff) / sw),
            "rms_difference_m": float(np.sqrt(np.sum(w * diff * diff) / sw)),
        }
    return result


def global_map_gate(hml: np.ndarray, base: np.ndarray, weights: np.ndarray,
                    wet: np.ndarray, nlev: int, label: str) -> None:
    """Enforce the preregistered all-wet-domain map contract."""
    require(hml.shape == wet.shape and base.shape == wet.shape,
            f"{label}: map shape mismatch")
    require(hml.dtype == np.float64, f"{label}: MLD is not float64")
    require(np.issubdtype(base.dtype, np.integer),
            f"{label}: base index is not integer")
    require(np.isfinite(hml[wet]).all(), f"{label}: non-finite wet MLD")
    require(np.isfinite(weights[wet]).all() and np.all(weights[wet] > 0.0),
            f"{label}: invalid wet area weight")
    require(np.all((base[wet] >= 0) & (base[wet] <= nlev - 2)),
            f"{label}: wet base index outside [0,{nlev - 2}]")


def difference_metrics(first: np.ndarray, second: np.ndarray,
                       weights: np.ndarray,
                       common_wet: np.ndarray) -> dict[str, dict[str, float]]:
    """Unscored regional sensitivity of one NEMO time level to another."""
    result: dict[str, dict[str, float]] = {}
    for name, rows in REGION_ROWS.items():
        select = np.zeros(common_wet.shape, dtype=bool)
        select[rows, :] = True
        wet = select & common_wet
        w = weights[wet]
        diff = first[wet] - second[wet]
        result[name] = {
            "mean_before_minus_now_m": float(np.sum(w * diff) / np.sum(w)),
            "rms_before_minus_now_m": float(
                np.sqrt(np.sum(w * diff * diff) / np.sum(w))),
            "max_abs_before_minus_now_m": float(np.max(np.abs(diff))),
            "changed_wet_columns": int(np.count_nonzero(diff)),
        }
    return result


def reduction_poison_control(weights: np.ndarray, wet: np.ndarray) -> dict[str, object]:
    base_lego = np.zeros(wet.shape, dtype=np.float64)
    base_nemo = np.zeros(wet.shape, dtype=np.float64)
    baseline = region_metrics(base_lego, base_nemo, weights, wet)

    dry_indices = np.argwhere(~wet)
    require(dry_indices.size > 0, "no dry cell available for poison control")
    dry = tuple(int(x) for x in dry_indices[0])
    dry_poison = base_lego.copy()
    dry_poison[dry] = 1.0e12
    after_dry = region_metrics(dry_poison, base_nemo, weights, wet)
    require(after_dry == baseline, "dry poison entered a regional reduction")

    eq_wet = np.argwhere(wet[99, :])
    require(eq_wet.size > 0, "no wet equatorial cell for poison control")
    wet_cell = (99, int(eq_wet[0, 0]))
    wet_poison = base_lego.copy()
    wet_poison[wet_cell] = 1.0e3
    after_wet = region_metrics(wet_poison, base_nemo, weights, wet)
    require(after_wet["equator"] != baseline["equator"],
            "wet poison failed to move equatorial reduction")
    return {
        "dry_cell": list(dry),
        "dry_poison_exactly_inert": True,
        "wet_cell": list(wet_cell),
        "wet_poison_moves_equator": True,
    }


def classify_day90(stats: dict[str, dict[str, float]]) -> str:
    matched = all(
        abs(stats[r]["mean_bias_m"]) <= H1 / 2.0
        and stats[r]["rms_difference_m"] <= H1
        for r in REGION_ROWS
    )
    diff = any(
        abs(stats[r]["mean_bias_m"]) >= H1
        or stats[r]["rms_difference_m"] >= 2.0 * H1
        for r in REGION_ROWS
    )
    if matched:
        return "MATCHED"
    if diff:
        return "DIFF"
    return "UNRESOLVED"


def tke_direction(prefix_rms: float, fix_rms: float) -> tuple[float, str]:
    delta = fix_rms - prefix_rms
    if delta <= -1.0:
        verdict = "TOWARD_NEMO"
    elif delta >= 1.0:
        verdict = "AWAY_FROM_NEMO"
    else:
        verdict = "INERT_AT_1M"
    return delta, verdict


def save_day90_figure(path: Path, biases: dict[str, np.ndarray]) -> None:
    limit = max(float(np.nanpercentile(np.abs(v), 99.0)) for v in biases.values())
    limit = max(limit, H1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    image = None
    for ax, (name, bias) in zip(axes.ravel(), biases.items()):
        image = ax.imshow(bias, origin="lower", aspect="auto", cmap="RdBu_r",
                          vmin=-limit, vmax=limit)
        ax.set_title(name)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label="hmlp legoESM - NEMO [m]")
    fig.suptitle("DINO day-90 exact zdf_mxl density-integral MLD bias")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-dir", default=None,
                        help="alternate writable git metadata used by this worktree")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/tmp/dino_mld_audit_codex"))
    args = parser.parse_args()

    require(jax.default_backend() == "cpu", "probe must run on CPU")
    require(bool(jax.config.x64_enabled), "JAX x64 must be enabled")
    set_policy(PrecisionPolicy.fp64())
    require(PREREG.is_file(), f"missing preregistration {PREREG}")
    repo_stamp = git_stamp(args.git_dir)

    arm_meta = {name: validate_arm(name, path, LOGS[name])
                for name, path in ARMS.items()}
    require(arm_meta["euc_tke_prefix"]["run_config"] ==
            arm_meta["euc_tke_floor_fix"]["run_config"],
            "EUC prefix/fix run-config stamps are not a controlled pair")

    grid = read_nemo_mesh_mask(str(MESH), nn_hls=0)
    initial = read_nemo_restart(str(restart_paths(0)[0]), nn_hls=0)
    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0,
        lon_east_deg=49.0,
        sill_lon_m_deg=1.0,
    )
    bridge = bridge_nemo_to_legoesm_topo(
        grid, initial, periodic_i=True, full_step=True, omega=cfg.omega,
        e3t_mode="both",
    )
    mc, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    z_coord = bridge.z_coord
    mask = jnp.asarray(bridge.state.land_mask.data, dtype=jnp.float64)
    h_bathy = jnp.asarray(bridge.state.H_bathy.data, dtype=jnp.float64)
    eos_fn = make_eos_fn(mc.eos, mc.eos_linear)
    active_3d = _nemo_native_active_3d(
        mask, z_coord, h_bathy, jnp.float64,
    )
    ladder_sha = VERTICAL_LADDER_SHA256(z_coord)
    for name, meta in arm_meta.items():
        require(meta["vertical_ladder_sha256"] == ladder_sha,
                f"{name}: saved ladder differs from freshly built bridge")

    executed_source_pairs = (
        (NEMO_ROOT / "src/OCE/ZDF/zdfmxl.F90", WORK / "zdfmxl.F90"),
        (NEMO_ROOT / "src/OCE/ZDF/zdfphy.F90", WORK / "zdfphy.F90"),
        (NEMO_ROOT / "src/OCE/DOM/domzgr.F90", WORK / "domzgr.F90"),
        (MY_SRC / "stpmlf.F90", WORK / "stpmlf.F90"),
    )
    for source, executed in executed_source_pairs:
        require(source.is_file() and executed.is_file(),
                f"missing oracle source pair: {source}, {executed}")
        require(sha256(source) == sha256(executed),
                f"executed DINO WORK source differs from quoted source: {executed}")

    resolved_h1 = float(np.asarray(jnp.cumsum(z_coord.dz_ref))[0])
    require(abs(resolved_h1 - H1) <= 1.0e-12,
            f"registered H1 {H1} differs from bridge {resolved_h1}")
    controls = {
        "synthetic": synthetic_controls(float(mc.g), float(mc.rho_0),
                                        float(mc.gm_redi.mld_rho_c)),
    }

    weights = np.asarray(grid.e1t * grid.e2t, dtype=np.float64)
    nemo_surface_wet = np.asarray(grid.tmask[..., 0]) > 0.5
    common_wet_by_arm = {
        name: nemo_surface_wet & (meta["land_mask"] > 0.5)
        for name, meta in arm_meta.items()
    }
    for name, common in common_wet_by_arm.items():
        require(np.array_equal(common, nemo_surface_wet),
                f"{name}: surface wet mask differs from NEMO")
    controls["reduction_poison"] = reduction_poison_control(
        weights, common_wet_by_arm["basin_legacy"])

    nemo_fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    nemo_online_fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    nemo_states: dict[int, dict[str, np.ndarray]] = {}
    for day in DAYS:
        pattern = str(RUN / f"DINO_{KT0 + STEPS_PER_DAY * day:08d}_restart*.nc")
        raw = REBUILD(pattern, ["tn", "sn", "tb", "sb", "sshn"])
        require(set(raw) == {"tn", "sn", "tb", "sb", "sshn"},
                f"day {day}: incomplete NEMO comparator state {sorted(raw)}")
        temp_n = np.moveaxis(raw["tn"], 0, -1)
        salt_n = np.moveaxis(raw["sn"], 0, -1)
        temp_b = np.moveaxis(raw["tb"], 0, -1)
        salt_b = np.moveaxis(raw["sb"], 0, -1)
        etan = raw["sshn"]
        require(np.isfinite(temp_n).all() and np.isfinite(salt_n).all()
                and np.isfinite(temp_b).all() and np.isfinite(salt_b).all()
                and np.isfinite(etan).all(),
                f"day {day}: incomplete stitched NEMO state")
        nemo_states[day] = {"T": temp_n, "S": salt_n, "eta": etan}
        nemo_fields[day] = compute_mld(
            temp_n, salt_n, etan, mask, h_bathy, z_coord, eos_fn, mc, active_3d)
        nemo_online_fields[day] = compute_mld(
            temp_b, salt_b, etan, mask, h_bathy, z_coord, eos_fn, mc, active_3d)

    day0_identity: dict[str, dict[str, float]] = {}
    nemo0 = nemo_states[0]
    for name, path in ARMS.items():
        common = common_wet_by_arm[name]
        active_np = np.asarray(active_3d) > 0.5
        with np.load(path, allow_pickle=False) as d:
            temp0 = np.asarray(d["T3d_day0"], dtype=np.float64)
            salt0 = np.asarray(d["S3d_day0"], dtype=np.float64)
        wet3 = active_np & common[..., None]
        max_t = float(np.max(np.abs(temp0[wet3] - nemo0["T"][wet3])))
        max_s = float(np.max(np.abs(salt0[wet3] - nemo0["S"][wet3])))
        max_mag = max(float(np.max(np.abs(temp0[wet3]))),
                      float(np.max(np.abs(salt0[wet3]))))
        fp32_quantum = float(np.spacing(np.float32(max_mag)))
        require(max_t <= 1.0e-5 and max_s <= 1.0e-5,
                f"{name}: day-0 T/S identity failed: {max_t}, {max_s}")
        day0_identity[name] = {
            "max_abs_T_difference": max_t,
            "max_abs_S_difference": max_s,
            "fp32_spacing_at_max_TS": fp32_quantum,
            "tolerance": 1.0e-5,
        }
    controls["day0_identity"] = day0_identity

    stats: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    maps: dict[str, np.ndarray] = {}
    day90_biases: dict[str, np.ndarray] = {}
    time_level_sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    nlev = int(np.asarray(z_coord.dz_ref).size)
    for day in DAYS:
        nemo_hml, nemo_base = nemo_fields[day]
        online_hml, online_base = nemo_online_fields[day]
        global_map_gate(nemo_hml, nemo_base, weights, nemo_surface_wet,
                        nlev, f"NEMO NOW day {day}")
        global_map_gate(online_hml, online_base, weights, nemo_surface_wet,
                        nlev, f"NEMO BEFORE/online day {day}")
        maps[f"nemo_hmlp_day{day}"] = np.where(nemo_surface_wet, nemo_hml, np.nan)
        maps[f"nemo_base_index_day{day}"] = np.where(
            nemo_surface_wet, nemo_base, -1).astype(np.int32)
        maps[f"nemo_online_time_level_hmlp_day{day}"] = np.where(
            nemo_surface_wet, online_hml, np.nan)
        maps[f"nemo_online_time_level_base_index_day{day}"] = np.where(
            nemo_surface_wet, online_base, -1).astype(np.int32)
        maps[f"nemo_before_minus_now_hmlp_day{day}"] = np.where(
            nemo_surface_wet, online_hml - nemo_hml, np.nan)
        time_level_sensitivity[str(day)] = difference_metrics(
            online_hml, nemo_hml, weights, nemo_surface_wet)
    for name, path in ARMS.items():
        stats[name] = {}
        common = common_wet_by_arm[name]
        with np.load(path, allow_pickle=False) as d:
            for day in DAYS:
                hml, base = compute_mld(
                    np.asarray(d[f"T3d_day{day}"], dtype=np.float64),
                    np.asarray(d[f"S3d_day{day}"], dtype=np.float64),
                    np.asarray(d[f"eta3d_day{day}"], dtype=np.float64),
                    mask, h_bathy, z_coord, eos_fn, mc, active_3d,
                )
                nemo_hml, _ = nemo_fields[day]
                global_map_gate(hml, base, weights, common, nlev,
                                f"{name} day {day}")
                bias = np.where(common, hml - nemo_hml, np.nan)
                require(bias.dtype == np.float64 and np.isfinite(bias[common]).all(),
                        f"{name} day {day}: invalid full-domain wet bias")
                stats[name][str(day)] = region_metrics(hml, nemo_hml, weights, common)
                maps[f"{name}_hmlp_day{day}"] = np.where(common, hml, np.nan)
                maps[f"{name}_base_index_day{day}"] = np.where(
                    common, base, -1).astype(np.int32)
                maps[f"{name}_bias_day{day}"] = bias
                if day == 90:
                    day90_biases[name] = bias

    classifications = {
        name: classify_day90(stats[name]["90"]) for name in ARMS
    }
    prefix_rms = stats["euc_tke_prefix"]["90"]["equator"]["rms_difference_m"]
    fix_rms = stats["euc_tke_floor_fix"]["90"]["equator"]["rms_difference_m"]
    delta_rms, direction = tke_direction(prefix_rms, fix_rms)

    source_inputs = [
        Path(__file__).resolve(), PREREG, MESH,
        ROOT / "packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py",
        ROOT / "scripts/validate/ocean_fidelity/rebuild_nemo_restart.py",
        ROOT / "scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py",
        NEMO_ROOT / "src/OCE/ZDF/zdfmxl.F90",
        NEMO_ROOT / "src/OCE/ZDF/zdfphy.F90",
        NEMO_ROOT / "src/OCE/DOM/domzgr.F90",
        WORK / "zdfmxl.F90", WORK / "zdfphy.F90", WORK / "domzgr.F90",
        WORK / "stpmlf.F90", MY_SRC / "stpmlf.F90",
        RUN / "namelist_cfg", RUN / "ocean.output",
        *ARMS.values(), *LOGS.values(),
    ]
    for day in DAYS:
        source_inputs.extend(restart_paths(day))
    for p in source_inputs:
        require(p.is_file(), f"provenance input vanished: {p}")
    hashes = {str(p): sha256(p) for p in sorted(set(source_inputs))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "mld_maps.npz", **maps)
    save_day90_figure(args.out_dir / "mld_bias_day90.png", day90_biases)
    artifact_hashes = {
        "mld_maps.npz": sha256(args.out_dir / "mld_maps.npz"),
        "mld_bias_day90.png": sha256(args.out_dir / "mld_bias_day90.png"),
    }
    result = {
        "scope": {
            "scored_time_level": "symmetric offline NOW T/S with NOW ssh",
            "native_online_time_level": (
                "NEMO BEFORE T/S-derived rn2b with NOW geometry; unscored because "
                "legoesm snapshots lack BEFORE T/S"),
            "claim": "exact zdf_mxl criterion, not native online hmlp",
        },
        "diagnostic": {
            "scored": (
                "symmetric offline NOW-state application of NEMO's exact "
                "zdf_mxl N2-integral density criterion"),
            "turbocline_hmld": "UNMEASURED_INPUT_HAS_NO_AVT",
            "dino_vertical_closures": {
                "TKE": True,
                "EVD": True,
                "DDM": False,
                "surface_wave_mixing": False,
                "internal_wave_mixing": False,
            },
            "rho_c_kg_m3": float(mc.gm_redi.mld_rho_c),
            "g_m_s2": float(mc.g),
            "rho0_kg_m3": float(mc.rho_0),
            "threshold_m_s2": float(mc.g * mc.gm_redi.mld_rho_c / mc.rho_0),
            "first_resolved_w_depth_m": resolved_h1,
        },
        "registered_thresholds": {
            "matched_abs_bias_max_m": H1 / 2.0,
            "matched_rms_max_m": H1,
            "diff_abs_bias_min_m": H1,
            "diff_rms_min_m": 2.0 * H1,
            "tke_direction_deadband_m": 1.0,
        },
        "controls": controls,
        "stats": stats,
        "post_review_unscored_nemo_time_level_sensitivity": time_level_sensitivity,
        "classifications_day90": classifications,
        "tke_floor_equatorial_day90": {
            "prefix_rms_m": prefix_rms,
            "fix_rms_m": fix_rms,
            "delta_rms_fix_minus_prefix_m": delta_rms,
            "classification": direction,
        },
        "provenance": {
            "audit_repo": repo_stamp,
            "nemo_repo": nemo_git_stamp(),
            "jax_backend": jax.default_backend(),
            "jax_x64": bool(jax.config.x64_enabled),
            "vertical_ladder_sha256": ladder_sha,
            "input_sha256": hashes,
            "producer_stamps": {
                name: {k: v for k, v in meta.items() if k != "land_mask"}
                for name, meta in arm_meta.items()
            },
            "artifact_sha256": artifact_hashes,
        },
    }
    result_path = args.out_dir / "mld_audit.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("SELF_CONTROLS: PASS")
    print("RETRACTION: native-online-hmlp wording withdrawn; the scored field is "
          "a symmetric offline NOW-state application of the exact criterion")
    for name in ARMS:
        print(f"DAY90_MLD {name}: {classifications[name]}")
        for region in REGION_ROWS:
            s = stats[name]["90"][region]
            print(f"  {region}: bias={s['mean_bias_m']:.9f} m "
                  f"rms={s['rms_difference_m']:.9f} m")
    print(f"TKE_FLOOR_EQUATOR_MLD: {direction} "
          f"delta_RMS={delta_rms:.9f} m "
          f"prefix={prefix_rms:.9f} m fix={fix_rms:.9f} m")
    print("POST_REVIEW_UNSCORED_NEMO_TIME_LEVEL_SENSITIVITY:")
    for day in DAYS:
        fields = []
        for region in REGION_ROWS:
            value = time_level_sensitivity[str(day)][region][
                "rms_before_minus_now_m"]
            fields.append(f"{region}={value:.9f} m")
        print(f"  day {day}: " + " ".join(fields))
    print("TURBOCLINE_HMLD: UNMEASURED_INPUT_HAS_NO_AVT")
    print(f"RESULT_JSON: {result_path}")
    print(f"MAPS_NPZ: {args.out_dir / 'mld_maps.npz'}")
    print(f"DAY90_MAP: {args.out_dir / 'mld_bias_day90.png'}")
    print(f"RESULT_JSON_SHA256: {sha256(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
