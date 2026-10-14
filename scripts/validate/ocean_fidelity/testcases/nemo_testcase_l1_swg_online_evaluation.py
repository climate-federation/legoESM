#!/usr/bin/env python3
"""Held-out SWG online-closure climate gate, no-closure controls, and plants.

Every candidate -- a trained network seed or a no-closure control -- is a
free run from rest scored on the SAME frozen two-year windows against the
coarse truth climatology made by the SAME operator as the training objective
(``symmetric`` = box restriction, ``template`` = filtered restriction).  The
bars are the truth's own window-to-window spread, frozen in ``FROZEN_GATE``
before any trained model exists; ``freeze --verify`` recomputes them from the
pinned data and refuses any drift.  Every day file is admitted by its own
per-file sampling label and, where the archive holds daily means beside the
snapshots, shown from the data not to be its own day's daily mean.  A
trained-closure score needs three or more DISTINCT runs (distinct day files
and distinct scored data), each manifest naming its checkpoint, network seed,
objective and training window.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
    build_swg_card,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemo_testcase_l1_swg_dg_climate as climate  # noqa: E402

DEFAULT_ROUND2 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2")
YEARS = (2, 3)
EXPECTED_CONTROL_MANIFEST_SHA256 = {
    "nu10": "cdbdf5607a4185275ba794d63be152eb643a03579350e0bf8c2758dcb924cd6a",
    "nu80": "cbee1f9f2268cec353c99bd3922b86a08a645f57cea81f9950791cbe35895f26",
    "truth": "aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c",
}
# A single-sampling archive (no daily mean beside its snapshot) holds nothing a
# daily mean could not also hold, so it is admitted only under one of these
# manifests, frozen in code, whose snapshot sampling the round-7 evaluation
# receipt shows from the data: bitwise one-day replay of both controls, and
# the midpoint test of the round-5 truth against its daily-mean twin.
FROZEN_SINGLE_SAMPLING_MANIFESTS = frozenset(
    EXPECTED_CONTROL_MANIFEST_SHA256.values())
# Registered 20-year inputs of the round-6 gate.  The fine truth is the
# 6.25 km run whose first 1095 days are bit-equal to the round-2 truth; only
# its END-OF-DAY SNAPSHOT fields are read, because both controls archive
# daily snapshots (a daily-mean truth would be a different quantity).
FINE_TRUTH_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/"
    "legoesm_runs/dg_gpu_7300d_truth")
CONTROL_ROOTS = {
    "nu10": DEFAULT_ROUND2.parent / "round1/dg50_cpu_7300d_no_closure",
    "nu80": DEFAULT_ROUND2 / "dg50_nu80_cpu_7300d",
}
# Per-file sampling labels exactly as the writers store them in EVERY day
# file (run_swg_double_gyre.py ``run_long``; swg_coarse_graining.py
# ``filter_truth_directory``).  The file's own label decides; a manifest never
# supplies one.
DUAL_SCHEMA = "legoesm-swg-double-gyre-daily-v2"  # snapshot AND daily mean
SINGLE_SAMPLING_SCHEMAS = (
    "legoesm-swg-double-gyre-daily-v1",  # driver: end-of-day snapshot only
    "legoesm-swg-conservative-truth-v2",  # restricted product, one sampling
)
SNAPSHOT_LABEL = "end-of-day instantaneous state"
DAILY_MEAN_LABEL = "arithmetic mean of every post-step state in model day"
SNAPSHOT_FIELDS = ("ssh_m", "u_m_s", "v_m_s")
DAILY_MEAN_FIELDS = ("ssh_daily_mean_m", "u_daily_mean_m_s", "v_daily_mean_m_s")
# What every trained-closure manifest must state under "closure".
CLOSURE_IDENTITY_KEYS = ("checkpoint_sha256", "network_seed", "objective",
                         "window_days")
GATE_WINDOW_DAYS = 730
GATE_FIRST_DAY = 366
SKILL_METRICS = ("mean_face_transport_rmse_m3_s", "ke_spectrum_relative_l1",
                 "wet_eke_map_nrmse")
NOISE_METRICS = ("top_octave_ke_fraction", "top_octave_enstrophy_fraction",
                 "native_grid_noise_index")
VARIABILITY_METRIC = "basin_eke_ratio"
SCORE_METRICS = SKILL_METRICS + (VARIABILITY_METRIC,) + NOISE_METRICS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _deck_geometry():
    card = build_swg_card(SWG_DG50_NAMELIST)
    mask = np.asarray(card.recipe.land_mask, dtype=bool)
    depth = np.asarray(
        card.recipe.z_coord.nemo_e3t_0[..., 0], dtype=np.float64)
    if not np.all(depth[mask] == depth[mask][0]):
        raise ValueError("online transport score requires one uniform layer")
    return mask, float(depth[mask][0])


@functools.lru_cache(maxsize=1)
def _dg50():
    """The dg50 card and C-grid geometry, built once per process."""
    from legoesm.ocean.fidelity.swg_coarse_graining import cgrid_geometry

    _deck_geometry()
    return build_swg_card(SWG_DG50_NAMELIST), cgrid_geometry(SWG_DG50_NAMELIST)


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


def window_statistics(arrays: tuple[np.ndarray, ...]) -> dict:
    """Diagnose one native dg50 (ssh, u, v) window once, candidate or truth."""
    from legoesm.ocean.fidelity.swg_coarse_graining import native_face_transports

    card, geometry = _dg50()
    days = int(arrays[0].shape[0])
    diagnosed = climate.diagnose(
        _native_array_provider("window", arrays), days=days, card=card,
        namelist=SWG_DG50_NAMELIST, diagnostics="online")
    ssh, u, v = arrays
    transports = [native_face_transports(
        {"T": ssh[day], "U": u[day], "V": v[day]}, geometry)
        for day in range(days)]
    health = {
        "all_finite": True,  # _native_array_provider refused anything else
        "minimum_total_layer_thickness_m": float(np.min(
            diagnosed["minimum_total_layer_thickness_m"])),
        "maximum_courant": float(max(
            np.max(diagnosed["maximum_u_courant"]),
            np.max(diagnosed["maximum_v_courant"]))),
    }
    health["passes"] = bool(
        health["minimum_total_layer_thickness_m"] > 0.0
        and health["maximum_courant"] < 1.0)
    return {
        "days": days,
        "mean_transport_m3_s": {kind: np.mean(np.stack([
            item[kind] for item in transports]), axis=0) for kind in ("U", "V")},
        "mean_kinetic_energy_spectrum_m3_s2": diagnosed[
            "mean_kinetic_energy_spectrum_m3_s2"],
        "eke_map_m2_s2": diagnosed["eke_map_m2_s2"],
        "basin_mean_eke_m2_s2": float(np.mean(diagnosed["basin_mean_eke_m2_s2"])),
        "top_octave_ke_fraction": float(diagnosed["top_octave_ke_fraction"][0]),
        "top_octave_enstrophy_fraction": float(
            diagnosed["top_octave_enstrophy_fraction"][0]),
        "native_grid_noise_index": float(np.mean(
            diagnosed["native_grid_noise_index"])),
        "physical_health": health,
    }


def score_statistics(candidate: dict, reference: dict) -> dict:
    """Score one candidate window against reference (truth) statistics."""
    _, geometry = _dg50()
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    face_error = np.concatenate([
        (candidate["mean_transport_m3_s"][kind]
         - reference["mean_transport_m3_s"][kind])[geometry["mask"][kind]]
        for kind in ("U", "V")])
    spectrum = "mean_kinetic_energy_spectrum_m3_s2"
    spectrum_l1 = float(np.sum(np.abs(candidate[spectrum] - reference[spectrum]))
                        / np.sum(np.abs(reference[spectrum])))
    candidate_eke = candidate["eke_map_m2_s2"]
    reference_eke = reference["eke_map_m2_s2"]
    eke_map_nrmse = float(np.sqrt(np.mean(
        (candidate_eke[wet] - reference_eke[wet]) ** 2))
        / np.sqrt(np.mean(reference_eke[wet] ** 2)))
    return {
        "mean_face_transport_rmse_m3_s": float(np.sqrt(np.mean(face_error**2))),
        "ke_spectrum_relative_l1": spectrum_l1,
        "wet_eke_map_nrmse": eke_map_nrmse,
        "basin_eke_ratio": (candidate["basin_mean_eke_m2_s2"]
                            / reference["basin_mean_eke_m2_s2"]),
        **{metric: candidate[metric] for metric in NOISE_METRICS},
        "physical_health": dict(candidate["physical_health"]),
    }


def score_native_trajectory(candidate_arrays: tuple[np.ndarray, ...],
                            truth_arrays: tuple[np.ndarray, ...]) -> dict:
    """Score arbitrary native model arrays through the control diagnostic."""
    if truth_arrays[0].shape[0] != candidate_arrays[0].shape[0]:
        raise ValueError("candidate and truth trajectories have different lengths")
    return score_statistics(window_statistics(candidate_arrays),
                            window_statistics(truth_arrays))


def climatology(window_stats: list[dict]) -> dict:
    """Equal-weight mean of equal-length per-window reference statistics."""
    if len(window_stats) < 2 or len({item["days"] for item in window_stats}) != 1:
        raise ValueError("a climatology needs two or more equal-length windows")

    def mean(values):
        return np.mean(np.stack(values), axis=0)

    spectrum = "mean_kinetic_energy_spectrum_m3_s2"
    return {
        "days": window_stats[0]["days"],
        "windows": len(window_stats),
        "mean_transport_m3_s": {kind: mean([
            item["mean_transport_m3_s"][kind] for item in window_stats])
            for kind in ("U", "V")},
        spectrum: mean([item[spectrum] for item in window_stats]),
        "eke_map_m2_s2": mean([item["eke_map_m2_s2"] for item in window_stats]),
        "basin_mean_eke_m2_s2": float(np.mean([
            item["basin_mean_eke_m2_s2"] for item in window_stats])),
    }


def leave_one_out_scores(window_stats: list[dict]) -> list[dict]:
    """Score each truth window against the climatology of all OTHER windows."""
    if len(window_stats) < 3:
        raise ValueError("truth spread needs at least three windows")
    return [score_statistics(window_stats[k], climatology(
        window_stats[:k] + window_stats[k + 1:]))
        for k in range(len(window_stats))]


def registered_windows(count: int) -> list[tuple[int, int]]:
    """The first ``count`` non-overlapping two-year windows after spin-up."""
    return [(GATE_FIRST_DAY + k * GATE_WINDOW_DAYS,
             GATE_FIRST_DAY + (k + 1) * GATE_WINDOW_DAYS - 1)
            for k in range(count)]


def truth_restrictor(objective: str):
    """Return the coarse-graining operator of one explicitly named objective."""
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        FILTER_TRUNCATE,
        cgrid_geometry,
        conservative_restrict_cgrid,
        filtered_restrict_cgrid,
    )
    from legoesm.training.swg_online import OBJECTIVE_OPERATORS

    if objective not in OBJECTIVE_OPERATORS:
        raise ValueError(f"objective must be one of {OBJECTIVE_OPERATORS}, "
                         f"got {objective!r}")
    source = cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = _dg50()[1]
    if objective == "symmetric":
        return functools.partial(
            conservative_restrict_cgrid, source=source, target=target)
    # Same sigma derivation as the driver's in-scan sampler: one dg50 cell.
    return functools.partial(
        filtered_restrict_cgrid, source=source, target=target,
        sigma_source_cells=float(target["spacing_m"]) / float(source["spacing_m"]),
        truncate=FILTER_TRUNCATE)


def _fine_truth_manifest(root: Path) -> dict:
    """The fine truth manifest; a still-running archive is read by its pins."""
    complete = root / "manifest.json"
    path = complete if complete.is_file() else root / "manifest.pending.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") not in ("complete", "pending"):
        raise ValueError(f"unusable fine truth manifest status in {path}")
    climate._require_legoesm_deck(
        manifest.get("namelist", {}), SWG_MADDISON_NAMELIST)
    return manifest


def fine_truth_pins_digest(root: Path, windows) -> str:
    """One SHA-256 over every pinned fine-truth day file the windows read."""
    pins = _fine_truth_manifest(root).get("daily_sha256", {})
    lines = []
    for first, last in windows:
        for day in range(first, last + 1):
            name = f"day{day:04d}.npz"
            if name not in pins:
                raise ValueError(f"fine truth does not pin {name} yet")
            lines.append(f"{name} {pins[name]}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _fields(saved, name: str, keys: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    if not set(keys) <= set(saved.files):
        raise ValueError(f"{name}: fields {keys} are absent")
    return tuple(np.asarray(saved[key], dtype=np.float64) for key in keys)


def _admit_day_file(saved, name: str):
    """(end-of-day snapshot, daily mean or None) of one day file, by its OWN label.

    The per-file ``schema`` is the writer's sampling label.  A dual (v2) file
    holds both samplings and must carry the driver's snapshot AND daily-mean
    labels; a single-sampling file must hold no daily-mean field.  A missing
    or unknown label refuses: nothing falls back to the manifest.
    """
    files = set(saved.files)
    schema = str(saved["schema"]) if "schema" in files else None
    if schema == DUAL_SCHEMA:
        labels = {key: str(saved[key]) if key in files else None
                  for key in ("snapshot_sampling", "daily_mean_sampling")}
        if labels != {"snapshot_sampling": SNAPSHOT_LABEL,
                      "daily_mean_sampling": DAILY_MEAN_LABEL}:
            raise ValueError(f"{name}: dual-file labels {labels} are not the "
                             "driver's snapshot and daily-mean labels")
        return (_fields(saved, name, SNAPSHOT_FIELDS),
                _fields(saved, name, DAILY_MEAN_FIELDS))
    if schema in SINGLE_SAMPLING_SCHEMAS:
        if any("daily_mean" in key for key in files):
            raise ValueError(f"{name}: a single-sampling {schema} file "
                             "carries daily-mean fields")
        return _fields(saved, name, SNAPSHOT_FIELDS), None
    raise ValueError(f"{name}: no known per-file sampling label "
                     f"(schema {schema!r})")


def _require_end_of_day_snapshots(name: str, snapshot, daily_mean,
                                  next_daily_mean) -> None:
    """Show from the data that a dual file's snapshot fields are end-of-day.

    The day-d end-of-day state lies half a day from the day-d and day-(d+1)
    daily means, so it is nearer their average than its own day's mean; a
    day's own daily mean put under its snapshot names IS that mean and fails.
    (The NEXT day's mean under those names passes: the test cannot see that
    shift.)  Per field, no parameter.  Only a non-finite SNAPSHOT field skips
    the test: it is then in the returned arrays, so a candidate's window
    fails physical health (``score_free_run``) and the truth refuses
    (``window_statistics``).  A finite snapshot beside a non-finite daily mean
    of its own or the next day refuses: renamed fields could hide behind it.
    """
    for field, s, m, n in zip(("ssh", "u", "v"), snapshot, daily_mean,
                              next_daily_mean, strict=True):
        if not np.all(np.isfinite(s)):
            continue
        for which, mean in (("its own day's", m), ("the next day's", n)):
            if not np.all(np.isfinite(mean)):
                raise ValueError(
                    f"{name}: the {field} snapshot is finite but {which} daily "
                    "mean is not, so its sampling cannot be shown from the "
                    "data (a renamed daily mean, or a run that blew up)")
        if not np.sum((s - 0.5 * (m + n)) ** 2) < np.sum((s - m) ** 2):
            raise ValueError(
                f"{name}: {field} is not an end-of-day snapshot -- it is no "
                "nearer the average of its own and the next day's daily means "
                "than its own day's mean")


def restrict_fine_truth_window(root: Path, first: int, last: int, *,
                               objective: str) -> tuple[np.ndarray, ...]:
    """Coarse truth (ssh, u, v) for days first..last from fine SNAPSHOTS.

    The fine truth is a dual archive: each day's snapshot is admitted by its
    own labels and shown from the data to be an end-of-day state (the check of
    day ``last`` reads day ``last + 1``'s daily mean).
    """
    pins = _fine_truth_manifest(root).get("daily_sha256", {})
    restrict = truth_restrictor(objective)
    values = {kind: [] for kind in "TUV"}
    previous = None
    for day in range(first, last + 2):
        name = f"day{day:04d}.npz"
        payload = (root / name).read_bytes()
        if name not in pins or hashlib.sha256(payload).hexdigest() != pins[name]:
            raise ValueError(f"fine truth digest differs or is unpinned: {name}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            if int(saved["day"]) != day:
                raise ValueError(f"{name}: embedded day differs")
            snapshot, daily_mean = _admit_day_file(saved, name)
        if daily_mean is None:
            raise ValueError(f"{name}: the fine truth must be a dual archive")
        if previous is not None:
            _require_end_of_day_snapshots(*previous, daily_mean)
        if day > last:
            break
        previous = (name, snapshot, daily_mean)
        coarse = restrict(dict(zip("TUV", snapshot, strict=True)))
        for kind in "TUV":
            values[kind].append(coarse[kind])
    return tuple(np.stack(values[kind]) for kind in "TUV")


def _day_file_pins(manifest: dict) -> dict:
    """The day-file digests a native manifest pins, under either writer's key
    (driver ``daily_sha256``; restricted product ``output_daily_sha256``).
    The ONE reading shared by the loader and the trained-run binding."""
    return manifest.get("daily_sha256", manifest.get("output_daily_sha256", {}))


def _read_admitted_day(root: Path, day: int, pins: dict):
    path = root / f"day{day:04d}.npz"
    if _sha256(path) != pins.get(path.name):
        raise ValueError(f"native trajectory digest differs: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return _admit_day_file(saved, path.name)


def _load_native_window(root: Path, start: int, days: int) -> tuple[np.ndarray, ...]:
    """End-of-day (ssh, u, v) of days start..start+days-1, admitted per file.

    A dual archive shows every snapshot from the data (the last day's check
    reads day start+days); a single-sampling archive is admitted only under a
    manifest frozen in this file.
    """
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete native trajectory {root}")
    pins = _day_file_pins(manifest)
    files = [_read_admitted_day(root, day, pins)
             for day in range(start, start + days)]
    means = [daily_mean for _, daily_mean in files]
    if any(daily_mean is None for daily_mean in means):
        if _sha256(manifest_path) not in FROZEN_SINGLE_SAMPLING_MANIFESTS:
            raise ValueError(
                f"{root}: a single-sampling archive cannot show from its data "
                "that its fields are end-of-day snapshots; only the frozen "
                "controls and the round-5 truth are admitted")
    else:
        means.append(_read_admitted_day(root, start + days, pins)[1])
        if means[-1] is None:
            raise ValueError(f"{root}: day {start + days} is not a dual file")
        for k, (snapshot, _) in enumerate(files):
            _require_end_of_day_snapshots(
                f"{root.name}/day{start + k:04d}.npz", snapshot, means[k],
                means[k + 1])
    return tuple(np.stack([snapshot[index] for snapshot, _ in files])
                 for index in range(3))


def score_control(root: Path, *, control: str) -> tuple[dict, dict]:
    """Round-5 record: one no-closure control on years 2--3 (not a gate)."""
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


# ------------------------------------------------------------------ gate ---
# Pre-registration, frozen BEFORE any trained model exists (round 6,
# 2026-10-14): per objective, the frozen windows, the pinned inputs, the
# truth's leave-one-out table and both controls' tables on the SAME windows.
# Every bar is ``gate_bars`` of the truth table (a test checks the stored
# bars equal that derivation) and ``freeze --verify`` recomputes both tables
# from the pinned data, refusing any difference.  Every tolerance and
# allowance ``held_out_gate`` applies is one of these truth bars (round 7:
# the noise allowance no longer uses a control's spread); the controls'
# tables supply only the levels a seed must beat or not exceed.
# Provenance of the frozen numbers: measured 2026-10-14 on the round-6 work
# copy (base a0b5535b2) by ``freeze`` with the functions below, BEFORE any
# trained model exists.  Truth: the fine truth's end-of-day snapshots, days
# 366-6205 (eight complete two-year windows when 6661 of 7300 days were
# pinned; ``fine_truth_pins_sha256`` digests every day file read), restricted
# by each objective's operator.  Controls: the frozen nu=10 and nu=80 runs,
# same windows, scored against the eight-window truth climatology.  Rule: the
# bar is the largest of eight leave-one-out values (8/9 coverage for a new
# exchangeable window), half_range = (bar - floor) / 2.  Re-verify with
# ``freeze --objective <name> --verify`` (exit 0 only on exact equality).
# Round 7 (2026-10-14): the ``template`` tables were re-measured the same way
# (same windows, pins and rule; still no trained model) because the round-6
# GPU fix evaluates the filtered restriction as matrix products, which moves
# 32 of its 189 numbers by at most 5.6e-15 relative; ``symmetric`` (box
# restriction) verified unchanged.
FROZEN_GATE: dict[str, dict] = {
    "symmetric": {
        "truth_operator": "box (exact volume/face-transport restriction)",
        "windows": [
            [366, 1095], [1096, 1825], [1826, 2555], [2556, 3285], [3286, 4015], [4016, 4745],
            [4746, 5475], [5476, 6205],
        ],
        "fine_truth_pins_sha256": (
            "f15177897479c7b662375de0c25960614919cb636926bf775ec7cbb0316b221b"),
        "control_manifest_sha256": {
            "nu10": "cdbdf5607a4185275ba794d63be152eb643a03579350e0bf8c2758dcb924cd6a",
            "nu80": "cbee1f9f2268cec353c99bd3922b86a08a645f57cea81f9950791cbe35895f26",
        },
        "truth_loo": {
            "mean_face_transport_rmse_m3_s": [
                869094.5660015474, 429039.93049046333, 935200.831748481, 341884.6539103485,
                576248.0785204493, 642946.9635092362, 522525.2571409132, 448621.84722457436,
            ],
            "ke_spectrum_relative_l1": [
                0.044034519168936824, 0.03199146297488733, 0.0366631686092397,
                0.057189579962744104, 0.03958609874700474, 0.05170498930444613,
                0.03621107415451166, 0.04058635720645127,
            ],
            "wet_eke_map_nrmse": [
                0.24208045768649325, 0.20768826927565262, 0.24765781167335535,
                0.20657687353381493, 0.254478861111718, 0.18826409656168017, 0.25858090941579265,
                0.223196663002778,
            ],
            "basin_eke_ratio": [
                0.999251688435746, 1.0047434093795409, 0.931070892402481, 1.0029680330158026,
                1.0160531362170242, 1.0262051244305066, 1.0215152241853473, 0.9989710848181484,
            ],
            "top_octave_ke_fraction": [
                0.008760401753645217, 0.008860422348663846, 0.008559049520095272,
                0.008074369011363469, 0.008638969160800333, 0.008663943236429437,
                0.008541656045203006, 0.009024889960101827,
            ],
            "top_octave_enstrophy_fraction": [
                0.12383097529317943, 0.125354594906669, 0.12271023427093879, 0.11650135295110754,
                0.12520805598005072, 0.1251041045474609, 0.12202750771970777, 0.128005739170692,
            ],
            "native_grid_noise_index": [
                0.489482533335054, 0.4926062532970408, 0.4838391559909701, 0.47614850062443664,
                0.48183976476974055, 0.4839031623267024, 0.48189755708230403, 0.4926887002587099,
            ],
        },
        "controls": {
            "nu10": {
                "mean_face_transport_rmse_m3_s": [
                    1092363.6477454659, 1026916.36872774, 1093104.5070973742, 1233002.2434098243,
                    1108228.1357472886, 1066923.67417058, 991275.3966122566, 1226118.7034810577,
                ],
                "ke_spectrum_relative_l1": [
                    0.19296069219922665, 0.1848398366359322, 0.19047948841264892,
                    0.23120342859911744, 0.2197628583811149, 0.19317243476634047,
                    0.19573207793989114, 0.2486829587218263,
                ],
                "wet_eke_map_nrmse": [
                    0.2892092430615287, 0.3165593778514286, 0.3071601709171936,
                    0.3289610943566974, 0.29772246387976875, 0.3098216930736585,
                    0.261502255890456, 0.32015288765137684,
                ],
                "basin_eke_ratio": [
                    0.7344257751230171, 0.7621170241379881, 0.7629273048538573,
                    0.6697243255206922, 0.7226722146247034, 0.7650266126925503,
                    0.7776403184069864, 0.6926066339781036,
                ],
                "top_octave_ke_fraction": [
                    0.05385808308751148, 0.05431220828775629, 0.053109340569998865,
                    0.053979415676443786, 0.05346221797822512, 0.05273034832293095,
                    0.05389270171069115, 0.05273495258351334,
                ],
                "top_octave_enstrophy_fraction": [
                    0.4358899035364099, 0.43520560133837277, 0.4321887687360149,
                    0.4382249162114404, 0.4316978435462325, 0.42787351577375754,
                    0.43224528212845337, 0.4331466508747122,
                ],
                "native_grid_noise_index": [
                    0.9317922221566435, 0.9324290909223998, 0.922547408868213, 0.934224007139235,
                    0.9298433834823261, 0.9245550241533739, 0.9312884462858325,
                    0.9220164021736723,
                ],
            },
            "nu80": {
                "mean_face_transport_rmse_m3_s": [
                    1556147.8888052863, 1474400.6004586369, 1413649.5152089477,
                    1175345.7945073915, 1300290.8857163473, 1527076.726261268, 1283369.659110044,
                    1528134.1438899613,
                ],
                "ke_spectrum_relative_l1": [
                    0.3543310060575042, 0.33886884207592705, 0.30385901465607534,
                    0.2342269661415031, 0.2814103123694682, 0.3526664483908289,
                    0.26965797793158036, 0.3472612090163635,
                ],
                "wet_eke_map_nrmse": [
                    0.7192582573364832, 0.6921375696197843, 0.6545250786663317,
                    0.5166506410225167, 0.6024572756460599, 0.7258322818092388,
                    0.6056155144780753, 0.7187004678632576,
                ],
                "basin_eke_ratio": [
                    0.39879593239265676, 0.4475947701656039, 0.45749121059436443,
                    0.5092609715333555, 0.4788804356771709, 0.4208966762379451,
                    0.48026989908589357, 0.41222320177540167,
                ],
                "top_octave_ke_fraction": [
                    0.027546136941835227, 0.027131413970082726, 0.027031429637570735,
                    0.029010706743404842, 0.028025033927312152, 0.02721218814779961,
                    0.028257450114799906, 0.02702600482533653,
                ],
                "top_octave_enstrophy_fraction": [
                    0.2947157212981263, 0.2925397160732689, 0.29027553872164685,
                    0.3019123477060896, 0.2980875159236257, 0.29336611927854683,
                    0.29838366084074874, 0.2918205856577802,
                ],
                "native_grid_noise_index": [
                    0.7069240384788047, 0.7037497725320242, 0.7005463310707691,
                    0.7204958190194521, 0.7089736016312511, 0.7036315574955214,
                    0.7117141617369971, 0.7018983509261939,
                ],
            },
        },
        "bars": {
            "mean_face_transport_rmse_m3_s": {
                "bar": 935200.831748481,
                "floor": 341884.6539103485,
                "half_range": 296658.08891906624,
            },
            "ke_spectrum_relative_l1": {
                "bar": 0.057189579962744104,
                "floor": 0.03199146297488733,
                "half_range": 0.012599058493928388,
            },
            "wet_eke_map_nrmse": {
                "bar": 0.25858090941579265,
                "floor": 0.18826409656168017,
                "half_range": 0.03515840642705624,
            },
            "basin_eke_ratio": {
                "bar": 1.0262051244305066,
                "floor": 0.931070892402481,
                "half_range": 0.04756711601401281,
            },
            "top_octave_ke_fraction": {
                "bar": 0.009024889960101827,
                "floor": 0.008074369011363469,
                "half_range": 0.00047526047436917927,
            },
            "top_octave_enstrophy_fraction": {
                "bar": 0.128005739170692,
                "floor": 0.11650135295110754,
                "half_range": 0.005752193109792227,
            },
            "native_grid_noise_index": {
                "bar": 0.4926887002587099,
                "floor": 0.47614850062443664,
                "half_range": 0.008270099817136639,
            },
        },
    },
    "template": {
        "truth_operator": "filtered (sigma 8 fine cells, then exact restriction)",
        "windows": [
            [366, 1095], [1096, 1825], [1826, 2555], [2556, 3285], [3286, 4015], [4016, 4745],
            [4746, 5475], [5476, 6205],
        ],
        "fine_truth_pins_sha256": (
            "f15177897479c7b662375de0c25960614919cb636926bf775ec7cbb0316b221b"),
        "control_manifest_sha256": {
            "nu10": "cdbdf5607a4185275ba794d63be152eb643a03579350e0bf8c2758dcb924cd6a",
            "nu80": "cbee1f9f2268cec353c99bd3922b86a08a645f57cea81f9950791cbe35895f26",
        },
        "truth_loo": {
            "mean_face_transport_rmse_m3_s": [
                754949.2910386896, 345043.6510893438, 796170.8494628742, 278936.75514474424,
                490331.20606561087, 541610.9793282589, 430041.96416013734, 369263.637561333,
            ],
            "ke_spectrum_relative_l1": [
                0.04634312114671689, 0.03307898609631619, 0.03978270749265124,
                0.061335289543215474, 0.03922056472970714, 0.05464462623367048,
                0.03768118350390047, 0.04096427759638369,
            ],
            "wet_eke_map_nrmse": [
                0.23655639317656227, 0.22171941944567466, 0.2458316605320523,
                0.21049654825158864, 0.2558334451448734, 0.17532772736041827, 0.2620022147484432,
                0.20579439039409086,
            ],
            "basin_eke_ratio": [
                0.994983117883836, 1.0049872450251598, 0.9265103620764747, 0.99941161796939,
                1.0246149593726162, 1.0378406595401541, 1.018203269237359, 0.9944352681221175,
            ],
            "top_octave_ke_fraction": [
                3.371815483302136e-05, 3.426831111280602e-05, 3.338981839532585e-05,
                3.133966791964047e-05, 3.3002868444745865e-05, 3.328678872345769e-05,
                3.314280891993964e-05, 3.494090681411433e-05,
            ],
            "top_octave_enstrophy_fraction": [
                0.0006988010060733777, 0.0007226455652212796, 0.0007003065542752054,
                0.0006483859202105985, 0.0006980134805603586, 0.0007060285543640957,
                0.000688190463908793, 0.0007379065533362526,
            ],
            "native_grid_noise_index": [
                0.21277639767314105, 0.21204275603340897, 0.21077828322537775,
                0.21128569759199586, 0.2083463319019164, 0.20873994350138503,
                0.21139078624502186, 0.2110783563435478,
            ],
        },
        "controls": {
            "nu10": {
                "mean_face_transport_rmse_m3_s": [
                    1365863.1183782492, 1303474.0380607818, 1357656.427005075,
                    1489055.9985645737, 1389740.3197816953, 1340529.728201074,
                    1275427.9362856902, 1484469.754389993,
                ],
                "ke_spectrum_relative_l1": [
                    0.4512402972018489, 0.4340509525168498, 0.44927314349116454,
                    0.449839887713791, 0.46251414796678675, 0.4423691406312488,
                    0.4457961472351256, 0.4805456988204709,
                ],
                "wet_eke_map_nrmse": [
                    0.4038072188813543, 0.4484507815524961, 0.4859794664264817,
                    0.3001141344786775, 0.3253127796201675, 0.4578763251629475,
                    0.4813048989550855, 0.3342603527588073,
                ],
                "basin_eke_ratio": [
                    1.0539613401925756, 1.0937005581122308, 1.0948633775259238,
                    0.961109443043614, 1.0370940150054087, 1.0978760567892234, 1.115977761045661,
                    0.993947436078967,
                ],
                "top_octave_ke_fraction": [
                    0.05385808308751148, 0.05431220828775629, 0.053109340569998865,
                    0.053979415676443786, 0.05346221797822512, 0.05273034832293095,
                    0.05389270171069115, 0.05273495258351334,
                ],
                "top_octave_enstrophy_fraction": [
                    0.4358899035364099, 0.43520560133837277, 0.4321887687360149,
                    0.4382249162114404, 0.4316978435462325, 0.42787351577375754,
                    0.43224528212845337, 0.4331466508747122,
                ],
                "native_grid_noise_index": [
                    0.9317922221566435, 0.9324290909223998, 0.922547408868213, 0.934224007139235,
                    0.9298433834823261, 0.9245550241533739, 0.9312884462858325,
                    0.9220164021736723,
                ],
            },
            "nu80": {
                "mean_face_transport_rmse_m3_s": [
                    1777032.89794184, 1705494.4490456928, 1653564.0528283883, 1437173.9490370022,
                    1553582.5254495614, 1754769.1613866945, 1538996.9481028898,
                    1755488.6780783804,
                ],
                "ke_spectrum_relative_l1": [
                    0.5671753904996489, 0.5610997959727322, 0.5230811346237294,
                    0.38093564699013255, 0.474086348814759, 0.5643374438065404,
                    0.47282542406462824, 0.5724012074733351,
                ],
                "wet_eke_map_nrmse": [
                    0.6304521698291154, 0.6249958721095124, 0.5777350474283582,
                    0.3777573605201305, 0.4810778213661539, 0.6696811997419511,
                    0.4841953735374593, 0.6400633240544512,
                ],
                "basin_eke_ratio": [
                    0.5723049348281779, 0.6423352771733938, 0.6565374824482364,
                    0.730831343678214, 0.6872327781439509, 0.6040213184185158,
                    0.6892267723173724, 0.5915741698048427,
                ],
                "top_octave_ke_fraction": [
                    0.027546136941835227, 0.027131413970082726, 0.027031429637570735,
                    0.029010706743404842, 0.028025033927312152, 0.02721218814779961,
                    0.028257450114799906, 0.02702600482533653,
                ],
                "top_octave_enstrophy_fraction": [
                    0.2947157212981263, 0.2925397160732689, 0.29027553872164685,
                    0.3019123477060896, 0.2980875159236257, 0.29336611927854683,
                    0.29838366084074874, 0.2918205856577802,
                ],
                "native_grid_noise_index": [
                    0.7069240384788047, 0.7037497725320242, 0.7005463310707691,
                    0.7204958190194521, 0.7089736016312511, 0.7036315574955214,
                    0.7117141617369971, 0.7018983509261939,
                ],
            },
        },
        "bars": {
            "mean_face_transport_rmse_m3_s": {
                "bar": 796170.8494628742,
                "floor": 278936.75514474424,
                "half_range": 258617.047159065,
            },
            "ke_spectrum_relative_l1": {
                "bar": 0.061335289543215474,
                "floor": 0.03307898609631619,
                "half_range": 0.014128151723449642,
            },
            "wet_eke_map_nrmse": {
                "bar": 0.2620022147484432,
                "floor": 0.17532772736041827,
                "half_range": 0.04333724369401247,
            },
            "basin_eke_ratio": {
                "bar": 1.0378406595401541,
                "floor": 0.9265103620764747,
                "half_range": 0.0556651487318397,
            },
            "top_octave_ke_fraction": {
                "bar": 3.494090681411433e-05,
                "floor": 3.133966791964047e-05,
                "half_range": 1.8006194472369296e-06,
            },
            "top_octave_enstrophy_fraction": {
                "bar": 0.0007379065533362526,
                "floor": 0.0006483859202105985,
                "half_range": 4.476031656282703e-05,
            },
            "native_grid_noise_index": {
                "bar": 0.21277639767314105,
                "floor": 0.2083463319019164,
                "half_range": 0.0022150328856123253,
            },
        },
    },
}


def gate_bars(table: dict[str, list[float]]) -> dict[str, dict]:
    """Window-to-window spread of one candidate's per-window values.

    ``bar`` is the largest of the K values -- for K exchangeable windows the
    K/(K+1) empirical quantile of a new window -- ``floor`` the smallest, and
    ``half_range`` = (bar - floor) / 2, this file's ``uncertainty_half_range``
    convention for sampling spread.
    """
    return {metric: {"bar": max(values), "floor": min(values),
                     "half_range": (max(values) - min(values)) / 2.0}
            for metric, values in table.items()}


def frozen_gate(objective: str) -> dict:
    """The frozen record of one explicitly selected objective."""
    from legoesm.training.swg_online import OBJECTIVE_OPERATORS

    if objective not in OBJECTIVE_OPERATORS:
        raise ValueError(f"objective must be one of {OBJECTIVE_OPERATORS}, "
                         f"got {objective!r}")
    if objective not in FROZEN_GATE:
        raise ValueError(f"no frozen gate for objective {objective!r}")
    return FROZEN_GATE[objective]


def _medians(rows: list[dict]) -> dict[str, float]:
    return {metric: float(np.median([row[metric] for row in rows]))
            for metric in SCORE_METRICS}


def held_out_gate(seed_scores: dict[str, list[dict]], *, objective: str) -> dict:
    """Apply the frozen gate to >= 3 seeds scored on the frozen windows.

    Every candidate is summarised by its median over the frozen windows.  The
    median seed must beat BOTH controls (frozen, verified scores on the same
    windows) by more than the truth's half-range on every skill measure;
    every seed's EKE ratio must lie inside the truth's own leave-one-out band
    [floor, bar]; no seed's noise may exceed the worse control's median by
    more than the truth's half-range; physical health is unconditional on
    every window and on the whole run.  Every tolerance and allowance is the
    TRUTH's own window-to-window spread (``bars``); a control's spread never
    enters (codex round-6 defect 2).
    """
    frozen = frozen_gate(objective)
    windows = len(frozen["windows"])
    if len(seed_scores) < 3:
        raise ValueError("held-out gate requires at least three network seeds")
    for name, rows in seed_scores.items():
        if len(rows) != windows:
            raise ValueError(f"{name}: needs one score per frozen window "
                             f"({windows}), got {len(rows)}")
    bars = frozen["bars"]
    healthy = {name: all(row.get("physical_health", {}).get("passes") is True
                         for row in rows)
               for name, rows in seed_scores.items()}
    result = {"objective": objective, "windows": frozen["windows"],
              "raw_seed_scores": seed_scores,
              "physical_health_by_seed": healthy,
              "physical_health_pass": all(healthy.values())}
    if not result["physical_health_pass"]:
        return result | {"pass": False}
    seeds = {name: _medians(rows) for name, rows in seed_scores.items()}
    controls = {name: {metric: float(np.median(values))
                       for metric, values in table.items()}
                for name, table in frozen["controls"].items()}
    median_seed = {metric: float(np.median([row[metric] for row in seeds.values()]))
                   for metric in SKILL_METRICS}
    required = {metric: min(row[metric] for row in controls.values())
                - bars[metric]["half_range"] for metric in SKILL_METRICS}
    noise_limit = {}
    for metric in NOISE_METRICS:
        worse = max(controls, key=lambda name: controls[name][metric])
        noise_limit[metric] = controls[worse][metric] + bars[metric]["half_range"]
    band = (bars[VARIABILITY_METRIC]["floor"], bars[VARIABILITY_METRIC]["bar"])
    skill_pass = all(median_seed[metric] < required[metric]
                     for metric in SKILL_METRICS)
    variability_pass = all(band[0] <= row[VARIABILITY_METRIC] <= band[1]
                           for row in seeds.values())
    noise_pass = all(row[metric] <= noise_limit[metric]
                     for row in seeds.values() for metric in NOISE_METRICS)
    return result | {
        "pass": bool(skill_pass and variability_pass and noise_pass),
        "seed_window_medians": seeds,
        "control_window_medians": controls,
        "median_seed_skill": median_seed,
        "skill_must_be_below": required,
        "eke_ratio_band": list(band),
        "noise_must_not_exceed": noise_limit,
        "skill_pass": bool(skill_pass),
        "variability_pass": bool(variability_pass),
        "spectral_health_pass": bool(noise_pass),
    }


def truth_as_seed(objective: str) -> list[dict]:
    """The frozen truth windows as one perfect-model seed (leave-one-out)."""
    table = frozen_gate(objective)["truth_loo"]
    return [{**{metric: table[metric][k] for metric in SCORE_METRICS},
             "physical_health": {"passes": True}}
            for k in range(len(table[SKILL_METRICS[0]]))]


def control_as_seed(objective: str, control: str) -> list[dict]:
    """One frozen control's windows as a seed (it must fail the gate)."""
    table = frozen_gate(objective)["controls"][control]
    return [{**{metric: table[metric][k] for metric in SCORE_METRICS},
             "physical_health": {"passes": True}}
            for k in range(len(table[SKILL_METRICS[0]]))]


def checkerboard_plant(arrays: tuple[np.ndarray, ...], amplitude_m_s: float, *,
                       rotational: bool) -> tuple[np.ndarray, ...]:
    """Add a day-to-day phase-flipped one-cell checkerboard on wet faces."""
    _, geometry = _dg50()
    ssh, u, v = arrays
    jj, ii = np.indices(u.shape[1:])
    checker = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    phase = np.where(np.arange(u.shape[0]) % 2 == 0, 1.0, -1.0)[:, None, None]
    sign_v = -1.0 if rotational else 1.0
    return (ssh,
            u + amplitude_m_s * phase * np.where(geometry["mask"]["U"], checker, 0.0),
            v + sign_v * amplitude_m_s * phase
            * np.where(geometry["mask"]["V"], checker, 0.0))


def collapsed_variance_plant(arrays: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Replace every day by the window's time mean (no eddies at all)."""
    return tuple(np.broadcast_to(np.mean(value, axis=0), value.shape).copy()
                 for value in arrays)


def plant_verdicts(truth_window: tuple[np.ndarray, ...], reference: dict, *,
                   objective: str) -> dict:
    """Score the mandatory plants through the real scorer and the gate."""
    windows = len(frozen_gate(objective)["windows"])
    verdicts = {}
    for name, arrays in (
            ("collapsed_variance", collapsed_variance_plant(truth_window)),
            ("native_checkerboard_0p1_m_s",
             checkerboard_plant(truth_window, 0.1, rotational=False))):
        score = score_statistics(window_statistics(arrays), reference)
        gate = held_out_gate({f"seed{k}": [score] * windows for k in range(3)},
                             objective=objective)
        verdicts[name] = {"score": score, "pass": gate["pass"],
                          "variability_pass": gate["variability_pass"],
                          "spectral_health_pass": gate["spectral_health_pass"]}
    return verdicts


def available_windows(root: Path = FINE_TRUTH_ROOT) -> list[tuple[int, int]]:
    """Every complete registered window the fine truth pins today, each with
    the following day pinned too (its last snapshot's sampling check)."""
    completed = int(_fine_truth_manifest(root)["completed_days"])
    return registered_windows(
        max(0, (completed - GATE_FIRST_DAY) // GATE_WINDOW_DAYS))


def freeze_record(objective: str, *, windows: list[tuple[int, int]],
                  fine_root: Path = FINE_TRUTH_ROOT,
                  control_roots: dict[str, Path] | None = None):
    """Measure the truth's leave-one-out table and the controls' tables.

    Returns the record, the first truth window's coarse arrays (for the
    plants) and the truth climatology every candidate is scored against.
    """
    control_roots = CONTROL_ROOTS if control_roots is None else control_roots
    truth, first_window = [], None
    for first, last in windows:
        arrays = restrict_fine_truth_window(
            fine_root, first, last, objective=objective)
        if first_window is None:
            first_window = arrays
        truth.append(window_statistics(arrays))
    reference = climatology(truth)
    loo = leave_one_out_scores(truth)
    controls = {}
    for name, root in control_roots.items():
        if _sha256(root / "manifest.json") != EXPECTED_CONTROL_MANIFEST_SHA256[name]:
            raise ValueError(f"{name} control manifest differs from frozen hash")
        controls[name] = [score_statistics(window_statistics(_load_native_window(
            root, first, last - first + 1)), reference) for first, last in windows]
    record = {
        "objective": objective,
        "windows": [list(window) for window in windows],
        "fine_truth_pins_sha256": fine_truth_pins_digest(fine_root, windows),
        "control_manifest_sha256": {
            name: EXPECTED_CONTROL_MANIFEST_SHA256[name] for name in controls},
        "truth_loo": {metric: [row[metric] for row in loo]
                      for metric in SCORE_METRICS},
        "controls": {name: {metric: [row[metric] for row in rows]
                            for metric in SCORE_METRICS}
                     for name, rows in controls.items()},
        "control_physical_health": {
            name: all(row["physical_health"]["passes"] for row in rows)
            for name, rows in controls.items()},
    }
    record["bars"] = gate_bars(record["truth_loo"])
    return record, first_window, reference


def score_free_run(root: Path, *, expected_manifest_sha256: str,
                   reference: dict, windows: list[tuple[int, int]]) -> list[dict]:
    """Score one bound dg50 free run; a non-finite window fails its health.

    The driver's whole-run health record (every day, not only the scored
    windows) must also be finite, positive-thickness and sub-unit Courant.
    Every row carries ``scored_data_sha256``, one SHA-256 over the window's
    loaded (ssh, u, v) arrays, so runs can be compared on their data.
    """
    if _sha256(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError(f"{root}: manifest differs from its bound SHA-256")
    manifest = json.loads((root / "manifest.json").read_text())
    climate._require_legoesm_deck(manifest.get("namelist", {}), SWG_DG50_NAMELIST)
    whole_run = manifest.get("numerical_health") or {}
    whole_run_ok = (
        whole_run.get("all_finite") is True
        and whole_run.get("minimum_total_layer_thickness_m", -1.0) > 0.0
        and max(whole_run.get("maximum_u_courant", 1.0),
                whole_run.get("maximum_v_courant", 1.0)) < 1.0)
    rows = []
    for first, last in windows:
        arrays = _load_native_window(root, first, last - first + 1)
        if all(np.all(np.isfinite(value)) for value in arrays):
            rows.append(score_statistics(window_statistics(arrays), reference))
        else:
            rows.append({"physical_health": {"all_finite": False,
                                             "passes": False}})
        if not whole_run_ok:
            rows[-1]["physical_health"] = dict(
                rows[-1]["physical_health"], passes=False, whole_run=whole_run)
        rows[-1]["scored_data_sha256"] = hashlib.sha256(
            b"".join(value.tobytes() for value in arrays)).hexdigest()
    return rows


def bind_trained_runs(seeds: dict[str, tuple[Path, str]], *, objective: str,
                      window_days: int) -> dict[str, dict]:
    """Refuse anything but three or more DISTINCT, identified trained runs.

    Each manifest, bound by its SHA-256, must state under ``closure`` the
    checkpoint SHA-256, the network seed, the objective and the training
    window length; objective and window must equal the score request.  Roots
    (as directories, so a symlink or bind mount is the same root), manifests,
    network seeds, checkpoints and day files must all differ between seeds;
    the day files are the non-empty set the loader reads (``_day_file_pins``).
    Identical data under different bytes is refused once the windows are
    loaded (``score_trained_closure``).
    """
    from legoesm.training.swg_online import ADMITTED_WINDOWS_DAYS

    frozen_gate(objective)
    if type(window_days) is not int or window_days not in ADMITTED_WINDOWS_DAYS:
        raise ValueError(f"window_days must be one of {ADMITTED_WINDOWS_DAYS}, "
                         f"got {window_days!r}")
    if len(seeds) < 3:
        raise ValueError("held-out gate requires at least three network seeds")
    bound = {}
    for label, (root, sha) in seeds.items():
        root = Path(root).resolve()
        if _sha256(root / "manifest.json") != sha:
            raise ValueError(f"{label}: manifest differs from its bound SHA-256")
        manifest = json.loads((root / "manifest.json").read_text())
        closure = manifest.get("closure")
        missing = (sorted(set(CLOSURE_IDENTITY_KEYS) - set(closure))
                   if isinstance(closure, dict) else list(CLOSURE_IDENTITY_KEYS))
        if missing:
            raise ValueError(f"{label}: manifest does not identify {missing}")
        checkpoint, network_seed = (closure["checkpoint_sha256"],
                                    closure["network_seed"])
        if not (isinstance(checkpoint, str) and len(checkpoint) == 64
                and set(checkpoint) <= set("0123456789abcdef")):
            raise ValueError(f"{label}: checkpoint_sha256 is not a SHA-256")
        if type(network_seed) is not int or network_seed < 0:
            raise ValueError(f"{label}: network_seed is not a non-negative int")
        request = {"objective": objective, "window_days": window_days}
        stated = {key: closure[key] for key in request}
        if stated != request:
            raise ValueError(f"{label}: manifest states {stated}, the score "
                             f"request {request}")
        pins = _day_file_pins(manifest)  # exactly the files the loader reads
        if not pins:
            raise ValueError(f"{label}: manifest pins no day files")
        directory = root.stat()
        bound[label] = {
            "root": str(root), "manifest_sha256": sha,
            "checkpoint_sha256": checkpoint, "network_seed": network_seed,
            "_directory": (directory.st_dev, directory.st_ino),
            "_day_files": set(pins.values()),
        }
    for key in ("_directory", "manifest_sha256", "network_seed",
                "checkpoint_sha256"):
        values = [item[key] for item in bound.values()]
        if len(set(values)) != len(values):
            raise ValueError(f"seeds share a {key.strip('_')}: three or more "
                             "distinct runs are required")
    labels = sorted(bound)
    for k, left in enumerate(labels):
        for right in labels[k + 1:]:
            if bound[left]["_day_files"] & bound[right]["_day_files"]:
                raise ValueError(f"{left} and {right} share day files: three "
                                 "or more distinct runs are required")
    return {label: {key: value for key, value in item.items()
                    if not key.startswith("_")}
            for label, item in bound.items()}


def score_trained_closure(seeds: dict[str, tuple[Path, str]], *, objective: str,
                          window_days: int,
                          fine_root: Path = FINE_TRUTH_ROOT) -> dict:
    """Score >= 3 distinct, explicitly bound trained-closure free runs.

    ``seeds`` maps a seed label to (free-run root, its manifest SHA-256); the
    runs are bound by ``bind_trained_runs`` before anything is computed.  The
    truth climatology and both controls are recomputed on the frozen windows
    and must reproduce the frozen tables exactly before any seed is scored.
    File bytes can differ while data do not, so two seeds whose loaded
    arrays are identical in any scored window are refused before the gate.
    """
    frozen = frozen_gate(objective)
    bindings = bind_trained_runs(seeds, objective=objective,
                                 window_days=window_days)
    windows = [tuple(window) for window in frozen["windows"]]
    record, _, reference = freeze_record(
        objective, windows=windows, fine_root=fine_root)
    for key in ("fine_truth_pins_sha256", "truth_loo", "controls"):
        if record[key] != frozen[key]:
            raise ValueError(f"gate inputs drifted from the frozen {key}")
    seed_scores = {name: score_free_run(
        Path(root), expected_manifest_sha256=sha, reference=reference,
        windows=windows) for name, (root, sha) in seeds.items()}
    first_seen = {}
    for name, rows in seed_scores.items():
        for window, row in zip(windows, rows, strict=True):
            other, where = first_seen.setdefault(
                row["scored_data_sha256"], (name, window))
            if other != name:
                raise ValueError(
                    f"{other} (window {where}) and {name} (window {window}) "
                    "hold identical data: three or more distinct runs are "
                    "required")
    return {"schema": "legoesm.swg_online_held_out_gate.v3",
            "objective": objective, "window_days": window_days,
            "seed_bindings": bindings,
            "gate": held_out_gate(seed_scores, objective=objective)}


def freeze_report(objective: str, *, verify: bool) -> dict:
    """Measure (or re-verify) one objective's frozen record and self-checks."""
    windows = ([tuple(window) for window in frozen_gate(objective)["windows"]]
               if verify else available_windows())
    record, first_window, reference = freeze_record(objective, windows=windows)
    report = {"record": record}
    if verify:
        frozen = frozen_gate(objective)
        report["verified_equal"] = {
            key: record[key] == frozen[key]
            for key in ("windows", "fine_truth_pins_sha256",
                        "control_manifest_sha256", "truth_loo", "controls",
                        "bars")}
        report["self_check"] = {
            "truth_passes": held_out_gate(
                {f"truth{k}": truth_as_seed(objective) for k in range(3)},
                objective=objective)["pass"],
            "controls_fail": {name: not held_out_gate(
                {f"{name}{k}": control_as_seed(objective, name)
                 for k in range(3)}, objective=objective)["pass"]
                for name in CONTROL_ROOTS},
            "plants_fail": {name: not value["pass"] for name, value in
                            plant_verdicts(first_window, reference,
                                           objective=objective).items()},
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser(
        "record", help="round-5 no-closure control record, years 2-3")
    record.add_argument("--round2-root", type=Path, default=DEFAULT_ROUND2)
    record.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser(
        "freeze", help="measure the truth-spread tables (--verify: the frozen)")
    freeze.add_argument("--objective", required=True)
    freeze.add_argument("--verify", action="store_true")
    freeze.add_argument("--output", type=Path, required=True)
    score = commands.add_parser(
        "score", help="score >= 3 trained-closure free runs through the gate")
    score.add_argument("--objective", required=True)
    score.add_argument("--window-days", type=int, required=True,
                       help="training window length every seed must state")
    score.add_argument("--seed", nargs=3, action="append", required=True,
                       metavar=("LABEL", "ROOT", "MANIFEST_SHA256"))
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            controls, provenance = {}, {}
            for name in ("nu10", "nu80"):
                controls[name], hashes = score_control(
                    args.round2_root, control=name)
                provenance.update(hashes)
            report = {"schema": "legoesm.swg_online_evaluation.v1",
                      "window": "years 2-3 conservative truth",
                      "controls": controls, "provenance": provenance}
            code = 0
        elif args.command == "freeze":
            report = freeze_report(args.objective, verify=args.verify)
            checks = report.get("self_check", {})
            code = 0 if (not args.verify or (
                all(report["verified_equal"].values())
                and checks["truth_passes"]
                and all(checks["controls_fail"].values())
                and all(checks["plants_fail"].values()))) else 1
        else:
            labels = [label for label, _, _ in args.seed]
            if len(set(labels)) != len(labels):
                raise ValueError("seed labels must be unique")
            report = score_trained_closure(
                {label: (Path(root), sha) for label, root, sha in args.seed},
                objective=args.objective, window_days=args.window_days)
            code = 0 if report["gate"]["pass"] else 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"{'PASS' if code == 0 else 'FAIL'} {args.command} -> {args.output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
