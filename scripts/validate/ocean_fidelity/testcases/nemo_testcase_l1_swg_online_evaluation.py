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
import nemo_testcase_l1_swg_dg_climate as climate  # noqa: E402
from nemo_testcase_l1_swg_dg_climate import grid_noise_index  # noqa: E402

DEFAULT_ROUND2 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2")
YEARS = (2, 3)
EXPECTED_CONTROL_MANIFEST_SHA256 = {
    "nu10": "cdbdf5607a4185275ba794d63be152eb643a03579350e0bf8c2758dcb924cd6a",
    "nu80": "cbee1f9f2268cec353c99bd3922b86a08a645f57cea81f9950791cbe35895f26",
    "truth": "aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c",
}
HELD_OUT_THRESHOLDS = {
    "mean_face_transport_rmse_m3_s": 1_439_733.0572095357,
    "ke_spectrum_relative_l1": 0.24850875740434986,
    "wet_eke_map_nrmse": 0.3936729545472666,
    "top_octave_ke_fraction": 0.005639566984841285,
    "top_octave_enstrophy_fraction": 0.06344752878701657,
    "native_grid_noise_index": 0.5644431624632231,
}


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


def _native_array_provider(name: str, arrays: tuple[np.ndarray, ...]):
    ssh, u, v = arrays
    if ssh.shape != u.shape or ssh.shape != v.shape or ssh.ndim != 3:
        raise ValueError("native trajectory arrays must share (day,y,x) shape")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("native trajectory contains non-finite fields")

    def iterate():
        for day in range(ssh.shape[0]):
            yield climate.DailyFrame(day + 1, ssh[day], u[day], v[day])

    return climate.DailyProvider(name, "daily_snapshot", {}, iterate)


def score_native_trajectory(candidate_arrays: tuple[np.ndarray, ...],
                            truth_arrays: tuple[np.ndarray, ...]) -> dict:
    """Score arbitrary native model arrays through the control diagnostic."""
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        native_face_transports,
    )

    card = build_swg_card(SWG_DG50_NAMELIST)
    days = int(candidate_arrays[0].shape[0])
    if truth_arrays[0].shape[0] != days:
        raise ValueError("candidate and truth trajectories have different lengths")
    diagnosed = [climate.diagnose(
        _native_array_provider(name, values), days=days, card=card,
        namelist=SWG_DG50_NAMELIST)
        for name, values in (("candidate", candidate_arrays),
                             ("truth", truth_arrays))]
    candidate, truth = diagnosed
    wet, _ = _deck_geometry()
    geometry = cgrid_geometry(SWG_DG50_NAMELIST)

    def mean_transport(arrays):
        ssh, u, v = arrays
        transports = [native_face_transports(
            {"T": ssh[day], "U": u[day], "V": v[day]}, geometry)
            for day in range(days)]
        return {kind: np.mean(np.stack([
            item[kind] for item in transports]), axis=0)
            for kind in ("U", "V")}

    candidate_q = mean_transport(candidate_arrays)
    truth_q = mean_transport(truth_arrays)
    face_error = np.concatenate([
        (candidate_q[kind] - truth_q[kind])[geometry["mask"][kind]]
        for kind in ("U", "V")])
    spectrum_l1 = float(np.sum(np.abs(
        candidate["mean_kinetic_energy_spectrum_m3_s2"]
        - truth["mean_kinetic_energy_spectrum_m3_s2"]))
        / np.sum(np.abs(truth["mean_kinetic_energy_spectrum_m3_s2"])))
    candidate_eke = candidate["eke_map_m2_s2"]
    truth_eke = truth["eke_map_m2_s2"]
    eke_map_nrmse = float(np.sqrt(np.mean(
        (candidate_eke[wet] - truth_eke[wet]) ** 2))
        / np.sqrt(np.mean(truth_eke[wet] ** 2)))
    eke_ratio = float(np.mean(candidate["basin_mean_eke_m2_s2"])
                      / np.mean(truth["basin_mean_eke_m2_s2"]))
    health = {
        "all_finite": bool(all(np.all(np.isfinite(value))
                                for value in candidate_arrays)),
        "minimum_total_layer_thickness_m": float(np.min(
            candidate["minimum_total_layer_thickness_m"])),
        "maximum_courant": float(max(
            np.max(candidate["maximum_u_courant"]),
            np.max(candidate["maximum_v_courant"]))),
    }
    health["passes"] = bool(
        health["all_finite"]
        and health["minimum_total_layer_thickness_m"] > 0.0
        and health["maximum_courant"] < 1.0)
    return {
        "mean_face_transport_rmse_m3_s": float(np.sqrt(np.mean(face_error**2))),
        "ke_spectrum_relative_l1": spectrum_l1,
        "wet_eke_map_nrmse": eke_map_nrmse,
        "basin_eke_ratio": eke_ratio,
        "top_octave_ke_fraction": float(candidate["top_octave_ke_fraction"][0]),
        "top_octave_enstrophy_fraction": float(
            candidate["top_octave_enstrophy_fraction"][0]),
        "native_grid_noise_index": float(np.mean(
            candidate["native_grid_noise_index"])),
        "physical_health": health,
    }


def _load_native_window(root: Path, start: int, days: int) -> tuple[np.ndarray, ...]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete native trajectory {root}")
    pins = manifest.get("daily_sha256", manifest.get("output_daily_sha256", {}))
    fields = ("ssh_m", "u_m_s", "v_m_s")
    values = {field: [] for field in fields}
    for day in range(start, start + days):
        path = root / f"day{day:04d}.npz"
        if _sha256(path) != pins.get(path.name):
            raise ValueError(f"native trajectory digest differs: {path}")
        with np.load(path, allow_pickle=False) as saved:
            for field in fields:
                values[field].append(np.asarray(saved[field], dtype=np.float64))
    return tuple(np.stack(values[field]) for field in fields)


def score_control(root: Path, *, control: str) -> tuple[dict, dict]:
    """Score one no-closure control from native outputs on years 2--3."""
    if control == "nu10":
        candidate_root = root.parent / "round1/dg50_cpu_7300d_no_closure"
    elif control == "nu80":
        candidate_root = root / "dg50_nu80_cpu_7300d"
    else:
        raise ValueError(f"unknown control {control!r}")
    truth_root = root / "conservative_truth_dg50_1095d/snapshots"
    expected = EXPECTED_CONTROL_MANIFEST_SHA256[control]
    if _sha256(candidate_root / "manifest.json") != expected:
        raise ValueError(f"{control} control manifest differs from frozen hash")
    if _sha256(truth_root / "manifest.json") \
            != EXPECTED_CONTROL_MANIFEST_SHA256["truth"]:
        raise ValueError("truth manifest differs from frozen hash")
    per_year = []
    for year in YEARS:
        start = (year - 1) * 365 + 1
        per_year.append(score_native_trajectory(
            _load_native_window(candidate_root, start, 365),
            _load_native_window(truth_root, start, 365)))
    scalar_keys = [key for key, value in per_year[0].items()
                   if key != "physical_health" and np.isscalar(value)]
    scores = {key: float(np.mean([row[key] for row in per_year]))
              for key in scalar_keys}
    scores["physical_health"] = {
        "passes": all(row["physical_health"]["passes"] for row in per_year),
        "per_year": [row["physical_health"] for row in per_year],
    }
    scores["per_year"] = per_year
    scores["uncertainty_half_range"] = {
        key: float(abs(per_year[1][key] - per_year[0][key]) / 2.0)
        for key in scalar_keys}
    provenance = {
        str(candidate_root / "manifest.json"): expected,
        str(truth_root / "manifest.json"):
            EXPECTED_CONTROL_MANIFEST_SHA256["truth"],
    }
    return scores, provenance


def held_out_gate(seed_scores: list[dict], controls: dict[str, dict]) -> dict:
    """Apply the pre-registered three-seed climate and spectral gate."""
    if len(seed_scores) < 3:
        raise ValueError("held-out gate requires at least three network seeds")
    skill = ("mean_face_transport_rmse_m3_s", "ke_spectrum_relative_l1",
             "wet_eke_map_nrmse")
    noise = ("top_octave_ke_fraction", "top_octave_enstrophy_fraction",
             "native_grid_noise_index")
    medians = {
        metric: float(np.median([seed[metric] for seed in seed_scores]))
        for metric in skill}
    better = {metric: HELD_OUT_THRESHOLDS[metric] for metric in skill}
    worse = {metric: HELD_OUT_THRESHOLDS[metric] for metric in noise}
    skill_pass = all(medians[metric] < better[metric] for metric in skill)
    variability_pass = all(
        0.8 <= seed["basin_eke_ratio"] <= 1.2 for seed in seed_scores)
    noise_pass = all(
        seed[metric] <= worse[metric]
        for seed in seed_scores for metric in noise)
    return {
        "pass": bool(skill_pass and variability_pass and noise_pass),
        "raw_seed_scores": seed_scores,
        "seed_uncertainty_half_range": {
            metric: 0.5 * (max(seed[metric] for seed in seed_scores)
                           - min(seed[metric] for seed in seed_scores))
            for metric in (*skill, "basin_eke_ratio", *noise)},
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
        "grid_noise_index": grid_noise_index(
            checkerboard, np.zeros_like(checkerboard)),
        "top_octave_fraction": 1.0,
    }


def adversarial_plants(controls: dict[str, dict]) -> dict:
    """Prove collapsed variance and two-cell noise cannot pass the gate."""
    measured = _physical_fail_plants()
    best = min(controls.values(), key=lambda value: sum((
        value["mean_face_transport_rmse_m3_s"], value["ke_spectrum_relative_l1"],
        value["wet_eke_map_nrmse"])))
    metric_names = (
        "mean_face_transport_rmse_m3_s", "ke_spectrum_relative_l1",
        "wet_eke_map_nrmse", "basin_eke_ratio", "top_octave_ke_fraction",
        "top_octave_enstrophy_fraction", "native_grid_noise_index")
    ideal = {name: best[name] for name in metric_names}
    for metric in metric_names[:3]:
        ideal[metric] = 0.0
    collapsed = dict(ideal, basin_eke_ratio=measured["collapsed_eke_ratio"])
    collapsed_gate = held_out_gate([collapsed] * 3, controls)
    noisy = dict(ideal, basin_eke_ratio=1.0)
    noisy["top_octave_ke_fraction"] = measured["top_octave_fraction"]
    noisy["top_octave_enstrophy_fraction"] = measured["top_octave_fraction"]
    noisy["native_grid_noise_index"] = measured["grid_noise_index"]
    noise_gate = held_out_gate([noisy] * 3, controls)
    if collapsed_gate["pass"] or noise_gate["pass"]:
        raise AssertionError("an adversarial closure passed the held-out gate")
    removes_noise_only = {name: best[name] for name in metric_names}
    for metric in metric_names[4:]:
        removes_noise_only[metric] = 0.0
    adds_eke_only = {name: best[name] for name in metric_names}
    adds_eke_only["basin_eke_ratio"] = 1.0
    removes_noise_gate = held_out_gate([removes_noise_only] * 3, controls)
    adds_eke_gate = held_out_gate([adds_eke_only] * 3, controls)
    if removes_noise_gate["pass"] or adds_eke_gate["pass"]:
        raise AssertionError("a one-sided closure passed the held-out gate")
    return {
        "collapsed_variance": {**collapsed_gate, "plant": measured},
        "two_cell_grid_noise": {**noise_gate, "plant": measured},
        "removes_grid_noise_only": removes_noise_gate,
        "adds_eddy_energy_only": adds_eke_gate,
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
