#!/usr/bin/env python
"""Real-path verification of the DINO EUC nemo_z0 surface-floor fix.

Runs the committed production closure on the shared day-180 matched state,
reconstructs the frozen pre-fix 47-column population from the removed clamp,
and appends per-column post-fix energy/mixing ratios to the campaign artifact.
No production physics is emulated for the post-fix result.
"""
from __future__ import annotations

import argparse
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

PINNED = {
    "kamm_twin_90d.py": "00b5f0d6c6c58ef984e010825239af27f768b83cf431e8763d236325c7cf68e6",
    "bn2_alpha_compare.py": "2ac9e4a457c4dded55a116d67938dfda035e79a3a98ccbb070d9a0ec3d11e40e",
    "dump_lane.py": "41252d898b0d09b350c086aa678ddb890c7e6232fa2b46e88093a3b108a9bf62",
    "zdftke_avm_offset_scan.py": "7ec40312dba6d70b7d6bb9ed5349b5861c7fa1b1ab4859b7c2f9c5c785300feb",
    "euc_mechanism.py": "7cf83a31ee4ede6fff6ae7dc095e215ce155809a74d0620288a6cbf55e5621d5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_sibling(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_euc_fix_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def ratio_summary(values: np.ndarray) -> dict[str, Any]:
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("ratio population contains non-positive/non-finite values")
    return {
        "geometric_mean": float(np.exp(np.mean(np.log(values)))),
        "median": float(np.median(values)),
        "iqr": np.quantile(values, [0.25, 0.75]).tolist(),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact",
        default="docs/ocean/fidelity/dino_euc_mechanism_artifact.json")
    args = ap.parse_args()

    observed_pins = {name: sha256(HERE / name) for name in PINNED}
    if observed_pins != PINNED:
        raise SystemExit(f"pinned sibling mismatch: expected={PINNED}, got={observed_pins}")
    kt = import_sibling("kamm_twin_90d")
    bac = import_sibling("bn2_alpha_compare")
    dl = import_sibling("dump_lane")
    zos = import_sibling("zdftke_avm_offset_scan")
    if dl.LANE != "d180":
        raise SystemExit(f"expected d180 lane, got {dl.LANE!r}")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, _dino_vertical_mixing_config, dino_config_for_recipe,
    )
    from legoesm.ocean.fidelity.precision_gate import require_fp64
    import legoesm.ocean.physics.vertical_mixing.tke as tke_module

    set_policy(PrecisionPolicy.fp64())
    br, cfg, mc, model, forcing, sf, state = kt._build_twin_state(
        "nemo_dino_kamm_mlf", dl.RUN_DIR, dl.RUN_DIR,
        bridge_tke=True, bridge_before=True, restart_file=dl.RESTART,
        e3t_mode="both")
    require_fp64(state, context="EUC surface-floor real-fix verification")
    module_path = Path(importlib.util.find_spec(
        "legoesm.ocean.physics.vertical_mixing").origin).resolve()
    if ROOT not in module_path.parents:
        raise SystemExit(f"vertical mixing imported outside worktree: {module_path}")

    conv_off = mc.physics.convection._replace(scheme="none")
    closure_model = LatLonCGridOceanModel(
        br.geometry, br.z_coord,
        mc._replace(physics=mc.physics._replace(convection=conv_off)))
    real_compute_k = tke_module.compute_K_from_tke
    real_solve = tke_module._solve_tke_backward_euler
    components: list[dict[str, Any]] = []
    solves: list[np.ndarray] = []

    def compute_spy(e, l_k, tke_cfg, *a, **kw):
        out = real_compute_k(e, l_k, tke_cfg, *a, **kw)
        components.append({
            "e": np.asarray(e), "l_k": np.asarray(l_k),
            "K_M": np.asarray(out[0]),
            "N2": None if kw.get("N2") is None else np.asarray(kw["N2"]),
            "floor": float(tke_cfg.kappaM_min),
            "surface_min": float(tke_cfg.tke_surface_min),
        })
        return out

    def solve_spy(*a, **kw):
        out = real_solve(*a, **kw)
        solves.append(np.asarray(out))
        return out

    tke_module.compute_K_from_tke = compute_spy
    tke_module._solve_tke_backward_euler = solve_spy
    try:
        K_fixed, A_fixed = zos.run_and_capture(closure_model, state, sf)
    finally:
        tke_module.compute_K_from_tke = real_compute_k
        tke_module._solve_tke_backward_euler = real_solve
    if A_fixed is None:
        raise SystemExit("production closure did not return momentum viscosity")
    shape_components = [c for c in components if c["K_M"].shape == A_fixed.shape]
    if not shape_components:
        raise SystemExit("no spied coefficient call matches production A_v shape")
    is_active = getattr(br.z_coord, "is_active", None)
    wet_interface = (
        np.ones_like(A_fixed, dtype=bool) if is_active is None
        else np.asarray(is_active[..., 1:]) > 0.5)
    if wet_interface.shape != A_fixed.shape:
        raise SystemExit(
            f"production wet mask {wet_interface.shape} != A_v {A_fixed.shape}")
    # compute_vertical_K_profiles applies this W-interface mask after the TKE
    # constructor returns. Compare like with like; an unmasked dry-column
    # value is not a different closure component.
    component_outputs = [c["K_M"] * wet_interface for c in shape_components]
    component_distances = [
        float(np.max(np.abs(out - A_fixed))) for out in component_outputs]
    component_pick = int(np.argmin(component_distances))
    component_pick_error = component_distances[component_pick]
    component_pick_tolerance = 1.0e-12 * max(
        1.0, float(np.max(np.abs(A_fixed))))
    if component_pick_error > component_pick_tolerance:
        raise SystemExit(
            "closest spied coefficient call is not the returned production "
            f"A_v: max_abs_error={component_pick_error}, "
            f"tolerance={component_pick_tolerance}")
    final = shape_components[component_pick]
    shape_solves = [s for s in solves if s.shape == A_fixed.shape]
    if len(shape_solves) != 1:
        raise SystemExit(f"expected one production TKE solve, got {len(shape_solves)}")
    solve_fixed = shape_solves[0]

    jpi, jpj, jpk, hls = bac._read_dims(dl.RUN_DIR)
    en_nemo = bac._load_interior(
        dl.dump_path("tke_dump_en.bin"), jpi - 2*hls, jpj - 2*hls)
    mxl_nemo = bac._load_interior(
        dl.dump_path("tke_dump_zmxlm.bin"), jpi - 2*hls, jpj - 2*hls)
    avm_nemo = bac._load_interior(
        dl.dump_path("tke_dump_avm_final.bin"), jpi - 2*hls, jpj - 2*hls)
    with nc.Dataset(Path(dl.RUN_DIR) / "mesh_mask.nc") as mm:
        def llz(name: str) -> np.ndarray:
            return np.moveaxis(np.asarray(mm[name][0]).squeeze(), 0, -1)
        tmask = llz("tmask") > 0.5
        gphit = np.asarray(mm["gphit"][0]).squeeze()
        glamt = np.asarray(mm["glamt"][0]).squeeze()
        gdepw = np.asarray(mm["gdepw_1d"][:]).squeeze()
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]
    row = int(np.nanargmin(np.abs(np.nanmean(gphit, axis=1))))
    if row != 99:
        raise SystemExit(f"equator row changed: {row}")

    # Freeze the original 47 columns without executing an emulated post-fix
    # result.  The removed legacy operation was exactly max(solve,jk2,1e-4).
    # At this level the previously proven buoyancy limb gives avm proportional
    # to en, so scaling the REAL fixed coefficient by legacy_en/fixed_en
    # reconstructs the old population.  Cross-check against the committed old
    # zonal-mean ratio before accepting the mask.
    k = 0
    etau_delta = final["e"] - solve_fixed
    legacy_en = np.maximum(solve_fixed[..., k], final["surface_min"]) \
        + etau_delta[..., k]
    fixed_en = final["e"][..., k]
    scale = legacy_en / fixed_en
    legacy_avm = A_fixed[..., k] * scale
    eligible = (
        wmask[row, :, 1]
        & np.isfinite(legacy_avm[row]) & np.isfinite(avm_nemo[row, :, 1])
        & (legacy_avm[row] > final["floor"] * (1.0 + 1e-12))
        & (avm_nemo[row, :, 1] > 1.2e-4 * (1.0 + 1e-12)))
    legacy_avm_ratio_all = np.divide(
        legacy_avm[row], avm_nemo[row, :, 1],
        out=np.full_like(legacy_avm[row], np.nan),
        where=np.abs(avm_nemo[row, :, 1]) > 0)
    population = eligible & (legacy_avm_ratio_all > 1.25)
    if int(np.count_nonzero(population)) != 47:
        raise SystemExit(f"frozen legacy population changed: {population.sum()} != 47")

    artifact_path = ROOT / args.artifact
    artifact = json.loads(artifact_path.read_text())
    prior_energy_ratio = artifact["tke_energy_single_root_10m"][
        "energy_ratio"]["geometric_mean"]
    reconstructed_energy_ratio = float(np.exp(np.mean(np.log(
        legacy_en[row, population] / en_nemo[row, population, 1]))))
    if abs(reconstructed_energy_ratio / prior_energy_ratio - 1.0) > 1.0e-10:
        raise SystemExit(
            "legacy cohort reconstruction failed old energy ratio: "
            f"{reconstructed_energy_ratio} vs {prior_energy_ratio}")

    idx = np.flatnonzero(population)
    en_ratio = fixed_en[row, idx] / en_nemo[row, idx, 1]
    mxl_ratio = final["l_k"][row, idx, k] / mxl_nemo[row, idx, 1]
    avm_ratio = A_fixed[row, idx, k] / avm_nemo[row, idx, 1]
    legacy_en_ratio = legacy_en[row, idx] / en_nemo[row, idx, 1]
    toward = np.abs(np.log(en_ratio)) < np.abs(np.log(legacy_en_ratio))
    replay_target = artifact["tke_equation_decomposition_10m"][
        "structural_surface_floor_replay"]["arm_over_nemo_geometric_mean"]
    en_stats = ratio_summary(en_ratio)
    match_relative_tolerance = 1.0e-10
    match_lower = replay_target * (1.0 - match_relative_tolerance)
    match_upper = replay_target * (1.0 + match_relative_tolerance)
    verdict = (
        "CONFIRM_REAL_FIX_MATCHES_OFFLINE_REPLAY"
        if (match_lower <= en_stats["geometric_mean"] <= match_upper
            and float(np.mean(toward)) == 1.0)
        else "REFUTE_REAL_FIX_MATCHES_OFFLINE_REPLAY")
    if verdict.startswith("REFUTE"):
        raise SystemExit(
            f"real fix missed replay: en={en_stats}, target={replay_target}, "
            f"toward={np.mean(toward)}")

    resolved_cards = {}
    for name in sorted(DINO_RECIPES):
        card = dino_config_for_recipe(name)
        resolved = _dino_vertical_mixing_config(card).tke
        resolved_cards[name] = {
            "vmix_scheme": card.vmix_scheme,
            "tke_surface_bc_level": resolved.tke_surface_bc_level,
        }
    nemo_z0_cards = [n for n, v in resolved_cards.items()
                     if v["tke_surface_bc_level"] == "nemo_z0"]
    if nemo_z0_cards != ["nemo_dino_kamm", "nemo_dino_kamm_mlf"]:
        raise SystemExit(f"unexpected shipped nemo_z0 cards: {nemo_z0_cards}")

    source_followup_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--",
         "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py"],
        cwd=ROOT, text=True).strip()
    clamp_fix_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H",
         "--grep=keep NEMO surface floor off interior TKE"],
        cwd=ROOT, text=True).strip()
    if not clamp_fix_commit:
        raise SystemExit("cannot resolve faithful surface-floor fix commit")
    repo_head = git_output("git", "rev-parse", "HEAD")
    repo_dirty = bool(git_output("git", "status", "--porcelain"))
    if repo_dirty:
        raise SystemExit("verification must run from a clean committed tree")
    per_column = [
        {
            "x_index_zero_based": int(i),
            "longitude_deg": float(glamt[row, i]),
            "legacy_en_over_nemo": float(le),
            "fixed_en_over_nemo": float(fe),
            "fixed_mxl_over_nemo": float(fm),
            "fixed_avm_over_nemo": float(fa),
            "energy_moved_toward_nemo": bool(mv),
        }
        for i, le, fe, fm, fa, mv in zip(
            idx, legacy_en_ratio, en_ratio, mxl_ratio, avm_ratio, toward)
    ]
    result = {
        "verdict": verdict,
        "fix_commit": clamp_fix_commit,
        "surface_knob_followup_commit": source_followup_commit,
        "verification_producer_head": repo_head,
        "state": "shared NEMO day-180 restart; first matched step kt=5761",
        "depth_m": float(gdepw[1]),
        "n_columns": int(idx.size),
        "energy_ratio_fixed_over_nemo": en_stats,
        "mixing_length_ratio_fixed_over_nemo": ratio_summary(mxl_ratio),
        "momentum_viscosity_ratio_fixed_over_nemo": ratio_summary(avm_ratio),
        "columns_energy_moved_toward_nemo_fraction": float(np.mean(toward)),
        "expected_replay_energy_ratio": float(replay_target),
        "replay_match_two_sided_band": {
            "lower": float(match_lower), "upper": float(match_upper),
            "relative_tolerance": match_relative_tolerance,
        },
        "spied_component_pick": {
            "matching_shape_calls": len(shape_components),
            "selected_zero_based": component_pick,
            "max_abs_error": component_pick_error,
            "assertion_tolerance": component_pick_tolerance,
        },
        "legacy_energy_ratio_reconstruction": reconstructed_energy_ratio,
        "per_column": per_column,
        "source_contract": {
            "nemo_surface": (
                "cfgs/DINO/MY_SRC/zdftke.F90:361 rn_emin0 at surface jk=1"),
            "nemo_interior": (
                "cfgs/DINO/MY_SRC/zdftke.F90:564-565 rn_emin at jk=2..jpkm1"),
            "lego_fix": (
                "tke.py: surface-min clamp guarded by "
                "surface_bc_level == 'interior_pinned'"),
        },
        "configuration_reachability": {
            "resolved_cards": resolved_cards,
            "shipped_nemo_z0_cards": nemo_z0_cards,
            "behavior_change": (
                "Only nemo_dino_kamm and inherited nemo_dino_kamm_mlf ship "
                "nemo_z0. Generic programmatic/audit overrides selecting "
                "nemo_z0 also receive the corrected interior floor. All "
                "other shipped cards remain interior_pinned and byte-pinned."),
        },
        "tests": {
            "focused_surface_terms": {
                "command": (
                    "env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "
                    "JAX_ENABLE_X64=1 PYTHONPATH=\"$PWD/packages/ocean:"
                    "$PWD/packages/core:$PWD/src:${PYTHONPATH:-}\" "
                    "/home/dbalwada/legoESM/.venv/bin/python -m pytest -q "
                    "tests/ocean/unit/test_tke_nemo_terms.py"),
                "result": "52 passed in 37.95s",
            },
            "focused_identity_and_prognostic": {
                "command": (
                    "env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "
                    "JAX_ENABLE_X64=1 PYTHONPATH=\"$PWD/packages/ocean:"
                    "$PWD/packages/core:$PWD/src:${PYTHONPATH:-}\" "
                    "/home/dbalwada/legoESM/.venv/bin/python -m pytest -q "
                    "tests/ocean/unit/test_tke_nemo_identity.py "
                    "tests/ocean/unit/test_tke_prognostic.py"),
                "result": "76 passed in 129.34s",
            },
            "vertical_mixing_consumers_non_mpas": {
                "command": (
                    "env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "
                    "JAX_ENABLE_X64=1 PYTHONPATH=\"$PWD/packages/ocean:"
                    "$PWD/packages/core:$PWD/src:${PYTHONPATH:-}\" "
                    "/home/dbalwada/legoESM/.venv/bin/python -m pytest -q "
                    "tests/ocean/unit/test_tke_integration.py "
                    "tests/ocean/unit/test_combined_pipeline_tke_evd_gate.py "
                    "tests/ocean/unit/test_implicit_vertical_mixing.py "
                    "tests/ocean/unit/test_tke_post_mixing.py "
                    "tests/ocean/unit/test_tke_dry_wmask.py "
                    "tests/ocean/unit/test_tke_n2_before_advection.py"),
                "result": "57 passed in 110.32s",
            },
            "vertical_mixing_consumers_mpas": {
                "command": (
                    "env -u JAX_ENABLE_X64 CUDA_VISIBLE_DEVICES='' "
                    "JAX_PLATFORMS=cpu PYTHONPATH=\"$PWD/packages/ocean:"
                    "$PWD/packages/core:$PWD/src:${PYTHONPATH:-}\" "
                    "/home/dbalwada/legoESM/.venv/bin/python -m pytest -q "
                    "tests/ocean/unit/test_mpas_tke.py"),
                "result": "36 passed, 9 warnings in 43.05s",
            },
            "mpas_forced_x64_diagnostic_not_a_pass": {
                "command": (
                    "env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "
                    "JAX_ENABLE_X64=1 PYTHONPATH=\"$PWD/packages/ocean:"
                    "$PWD/packages/core:$PWD/src:${PYTHONPATH:-}\" "
                    "/home/dbalwada/legoESM/.venv/bin/python -m pytest -q "
                    "tests/ocean/unit/test_mpas_tke.py"),
                "result": "34 passed, 2 failed, 9 warnings in 39.41s",
                "cause": (
                    "forced process-global x64 makes scan inputs float64 "
                    "against the MPAS float32 policy; the supported clean "
                    "policy command above passes"),
            },
            "red_before_fix": {
                "result": (
                    "reviewer reproduction at 2cb678259: 2 failed; the prior "
                    "report's 1-failed count is retracted"),
            },
        },
        "provenance": {
            "repo": {"head": repo_head, "dirty": repo_dirty,
                     "branch": git_output("git", "branch", "--show-current")},
            "pinned_reducers": observed_pins,
            "input_sha256": {
                "restart": sha256(Path(dl.restart_path())),
                "nemo_en": sha256(Path(dl.dump_path("tke_dump_en.bin"))),
                "nemo_zmxlm": sha256(Path(dl.dump_path("tke_dump_zmxlm.bin"))),
                "nemo_avm": sha256(Path(dl.dump_path("tke_dump_avm_final.bin"))),
                "mesh_mask": sha256(Path(dl.RUN_DIR) / "mesh_mask.nc"),
                "fixed_tke_source": sha256(
                    ROOT / "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py"),
                "regression_test": sha256(
                    ROOT / "tests/ocean/unit/test_tke_nemo_terms.py"),
                "verifier": sha256(Path(__file__).resolve()),
            },
        },
    }
    artifact["schema"] = "dino-euc-mechanism-v3-faithful-fix"
    artifact["faithful_surface_floor_fix"] = result
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    reread = json.loads(artifact_path.read_text())
    if reread["faithful_surface_floor_fix"] != result:
        raise SystemExit("artifact read-back mismatch")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
