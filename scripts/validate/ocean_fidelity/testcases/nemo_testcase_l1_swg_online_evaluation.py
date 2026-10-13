#!/usr/bin/env python3
"""Held-out SWG online-closure scorecard and adversarial gate plants."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    build_swg_card,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_l1_swg_dg_climate import checkerboard_index  # noqa: E402

DEFAULT_ROUND2 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2")
YEARS = (2, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_year(root: Path, directory: str):
    path = root / directory / "swg_dg_climate_arrays.npz"
    if not path.is_file():
        raise ValueError(f"missing admitted climate arrays {path}")
    with np.load(path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return path, arrays


def _deck_geometry():
    card = build_swg_card(SWG_DG50_NAMELIST)
    mask = np.asarray(card.recipe.land_mask, dtype=bool)
    depth = np.asarray(
        card.recipe.z_coord.nemo_e3t_0[..., 0], dtype=np.float64)
    if not np.all(depth[mask] == depth[mask][0]):
        raise ValueError("online transport score requires one uniform layer")
    return mask, float(depth[mask][0])


def _mean(years, key):
    return np.mean(np.stack([year[key] for year in years]), axis=0)


def score_control(root: Path, *, control: str) -> tuple[dict, dict]:
    """Score one round-2 no-closure control on the common years 2--3."""
    if control == "nu10":
        directories = [f"climate_nu10_v2_year{year}" for year in YEARS]
        candidate = "dg50 no closure"
    elif control == "nu80":
        directories = [f"climate_nu80_year{year}" for year in YEARS]
        candidate = "dg50 nu80 no closure"
    else:
        raise ValueError(f"unknown control {control!r}")
    loaded = [_load_year(root, directory) for directory in directories]
    paths = [item[0] for item in loaded]
    years = [item[1] for item in loaded]
    truth = "filtered truth"
    wet, layer_depth_m = _deck_geometry()

    mean_u = _mean(years, f"{candidate}.mean_u_m_s")
    mean_v = _mean(years, f"{candidate}.mean_v_m_s")
    truth_u = _mean(years, f"{truth}.mean_u_m_s")
    truth_v = _mean(years, f"{truth}.mean_v_m_s")
    # Single-layer transport uses the deck's 500 m reference depth.  This is
    # an absolute held-out error, not the old "lies between" criterion.
    transport_error = float(layer_depth_m * np.sqrt(np.mean(
        ((mean_u - truth_u) ** 2 + (mean_v - truth_v) ** 2)[wet])))
    spectrum = _mean(years, f"{candidate}.mean_kinetic_energy_spectrum_m3_s2")
    truth_spectrum = _mean(
        years, f"{truth}.mean_kinetic_energy_spectrum_m3_s2")
    spectrum_l1 = float(
        np.sum(np.abs(spectrum - truth_spectrum))
        / np.sum(np.abs(truth_spectrum)))
    eke_map = _mean(years, f"{candidate}.eke_map_m2_s2")
    truth_eke_map = _mean(years, f"{truth}.eke_map_m2_s2")
    eke_map_nrmse = float(
        np.sqrt(np.mean((eke_map[wet] - truth_eke_map[wet]) ** 2))
        / np.sqrt(np.mean(truth_eke_map[wet] ** 2)))
    eke = float(np.mean(np.concatenate([
        year[f"{candidate}.basin_mean_eke_m2_s2"] for year in years])))
    truth_eke = float(np.mean(np.concatenate([
        year[f"{truth}.basin_mean_eke_m2_s2"] for year in years])))
    scores = {
        "mean_transport_rmse_m2_s": transport_error,
        "ke_spectrum_relative_l1": spectrum_l1,
        "wet_eke_map_nrmse": eke_map_nrmse,
        "basin_eke_ratio": eke / truth_eke,
        "top_octave_ke_fraction": float(np.mean([
            year[f"{candidate}.top_octave_ke_fraction"] for year in years])),
        "top_octave_enstrophy_fraction": float(np.mean([
            year[f"{candidate}.top_octave_enstrophy_fraction"]
            for year in years])),
        "native_checkerboard_index": float(np.mean(np.concatenate([
            year[f"{candidate}.native_checkerboard_index"] for year in years]))),
    }
    provenance = {str(path): _sha256(path) for path in paths}
    return scores, provenance


def held_out_gate(seed_scores: list[dict], controls: dict[str, dict]) -> dict:
    """Apply the pre-registered three-seed climate and spectral gate."""
    if len(seed_scores) < 3:
        raise ValueError("held-out gate requires at least three network seeds")
    skill = ("mean_transport_rmse_m2_s", "ke_spectrum_relative_l1",
             "wet_eke_map_nrmse")
    noise = ("top_octave_ke_fraction", "top_octave_enstrophy_fraction",
             "native_checkerboard_index")
    medians = {
        metric: float(np.median([seed[metric] for seed in seed_scores]))
        for metric in skill}
    better = {metric: min(control[metric] for control in controls.values())
              for metric in skill}
    worse = {metric: max(control[metric] for control in controls.values())
             for metric in noise}
    skill_pass = all(medians[metric] < better[metric] for metric in skill)
    variability_pass = all(
        0.8 <= seed["basin_eke_ratio"] <= 1.2 for seed in seed_scores)
    noise_pass = all(
        seed[metric] <= worse[metric]
        for seed in seed_scores for metric in noise)
    return {
        "pass": bool(skill_pass and variability_pass and noise_pass),
        "median_skill": medians,
        "better_control_skill": better,
        "worse_control_noise": worse,
        "skill_pass": bool(skill_pass),
        "variability_pass": bool(variability_pass),
        "spectral_health_pass": bool(noise_pass),
    }


def _physical_fail_plants() -> dict:
    """Measure collapse and grid-mode plants before injecting their scores."""
    jj, ii = np.indices((80, 80))
    checkerboard = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    anomaly = np.stack((-checkerboard, checkerboard))
    collapsed = np.broadcast_to(np.mean(anomaly, axis=0), anomaly.shape)
    source_energy = np.mean(anomaly * anomaly, dtype=np.float64)
    collapsed_energy = np.mean(collapsed * collapsed, dtype=np.float64)
    return {
        "collapsed_eke_ratio": float(collapsed_energy / source_energy),
        "checkerboard_index": checkerboard_index(
            checkerboard, np.zeros_like(checkerboard)),
        "top_octave_fraction": 1.0,
    }


def adversarial_plants(controls: dict[str, dict]) -> dict:
    """Prove collapsed variance and two-cell noise cannot pass the gate."""
    measured = _physical_fail_plants()
    best = min(controls.values(), key=lambda value: sum((
        value["mean_transport_rmse_m2_s"], value["ke_spectrum_relative_l1"],
        value["wet_eke_map_nrmse"])))
    ideal = dict(best)
    for metric in ("mean_transport_rmse_m2_s", "ke_spectrum_relative_l1",
                   "wet_eke_map_nrmse"):
        ideal[metric] = 0.0
    collapsed = dict(ideal, basin_eke_ratio=measured["collapsed_eke_ratio"])
    collapsed_gate = held_out_gate([collapsed] * 3, controls)
    noisy = dict(ideal, basin_eke_ratio=1.0)
    noisy["top_octave_ke_fraction"] = measured["top_octave_fraction"]
    noisy["top_octave_enstrophy_fraction"] = measured["top_octave_fraction"]
    noisy["native_checkerboard_index"] = measured["checkerboard_index"]
    noise_gate = held_out_gate([noisy] * 3, controls)
    if collapsed_gate["pass"] or noise_gate["pass"]:
        raise AssertionError("an adversarial closure passed the held-out gate")
    return {
        "collapsed_variance": {**collapsed_gate, "plant": measured},
        "two_cell_grid_noise": {**noise_gate, "plant": measured},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round2-root", type=Path, default=DEFAULT_ROUND2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controls = {}
    provenance = {}
    for name in ("nu10", "nu80"):
        controls[name], hashes = score_control(args.round2_root, control=name)
        provenance.update(hashes)
    report = {
        "schema": "legoesm.swg_online_evaluation.v1",
        "window": "years 2-3 conservative truth",
        "controls": controls,
        "plants": adversarial_plants(controls),
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
