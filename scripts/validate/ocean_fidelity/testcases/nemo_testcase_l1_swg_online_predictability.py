#!/usr/bin/env python3
"""Measure fixed-seed SWG twin divergence and the first SSH e-folding time."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_efold_crossing(days: np.ndarray, rms: np.ndarray,
                         initial_rms: float) -> float | None:
    """Log-interpolate the first crossing of e times the initial RMS."""
    days = np.asarray(days, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    if days.ndim != 1 or rms.shape != days.shape or days.size < 2:
        raise ValueError("e-folding curve must have at least two paired points")
    if not np.all(np.isfinite(days)) or not np.all(np.isfinite(rms)) \
            or initial_rms <= 0.0 or np.any(rms <= 0.0):
        raise ValueError("e-folding inputs must be positive finite")
    threshold = np.e * initial_rms
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0:
        return None
    hi = int(above[0])
    if hi == 0:
        return float(days[0])
    lo = hi - 1
    fraction = ((np.log(threshold) - np.log(rms[lo]))
                / (np.log(rms[hi]) - np.log(rms[lo])))
    return float(days[lo] + fraction * (days[hi] - days[lo]))


def measure_twins(control_root: Path, perturbed_root: Path) -> dict:
    """Compare two complete fine-deck daily runs from one restart."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
    )

    manifests = []
    for root in (control_root, perturbed_root):
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete" \
                or manifest.get("sampling") != "daily_snapshot" \
                or manifest.get("completed_days") != manifest.get("days"):
            raise ValueError(f"incomplete daily-snapshot twin: {root}")
        manifests.append((manifest, sha256(path)))
    control, perturbed = (item[0] for item in manifests)
    if control["days"] != perturbed["days"] \
            or control["namelist"] != perturbed["namelist"]:
        raise ValueError("twin clocks or resolved decks differ")
    expected = json.loads(json.dumps(SWG_MADDISON_NAMELIST._asdict()))
    for field in expected.keys() - {"nn_itend", "nn_stock"}:
        if control["namelist"].get(field) != expected[field]:
            raise ValueError(f"twins differ from fine deck at {field}")
    c_restart, p_restart = control.get("restart"), perturbed.get("restart")
    if c_restart is None or p_restart is None:
        raise ValueError("both twins must declare their restart")
    for field in ("path", "sha256", "source_day", "source_step"):
        if c_restart.get(field) != p_restart.get(field):
            raise ValueError(f"twin restart provenance differs at {field}")
    if c_restart.get("perturb_amplitude_m") not in (None, 0.0):
        raise ValueError("control twin is perturbed")
    amplitude = p_restart.get("perturb_amplitude_m")
    seed = p_restart.get("perturb_seed")
    if not isinstance(amplitude, (int, float)) or amplitude <= 0.0 or seed != 0:
        raise ValueError("perturbed twin must use positive amplitude and fixed seed 0")
    card = build_swg_card(SWG_MADDISON_NAMELIST)
    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    umask = np.asarray(u_full[:, 1:], dtype=bool)
    vmask = np.asarray(v_full[1:, :], dtype=bool)
    noise = np.random.default_rng(seed).standard_normal(tmask.shape)
    initial_ssh_rms = float(amplitude * np.sqrt(
        np.mean(noise[tmask] ** 2, dtype=np.float64)))
    days = int(control["days"])
    lead = np.arange(days + 1, dtype=np.float64)
    ssh_rms = np.empty(days + 1, dtype=np.float64)
    u_rms = np.empty(days + 1, dtype=np.float64)
    v_rms = np.empty(days + 1, dtype=np.float64)
    ssh_rms[0], u_rms[0], v_rms[0] = initial_ssh_rms, 0.0, 0.0
    for day in range(1, days + 1):
        name = f"day{day:04d}.npz"
        paths = (control_root / name, perturbed_root / name)
        for manifest, path in zip((control, perturbed), paths, strict=True):
            if sha256(path) != manifest["daily_sha256"].get(name):
                raise ValueError(f"twin daily digest mismatch: {path}")
        with np.load(paths[0], allow_pickle=False) as left, \
                np.load(paths[1], allow_pickle=False) as right:
            delta_ssh = np.asarray(right["ssh_m"] - left["ssh_m"])
            delta_u = np.asarray(right["u_m_s"] - left["u_m_s"])
            delta_v = np.asarray(right["v_m_s"] - left["v_m_s"])
        if not all(np.all(np.isfinite(value))
                   for value in (delta_ssh, delta_u, delta_v)):
            raise ValueError(f"non-finite twin difference at lead day {day}")
        ssh_rms[day] = np.sqrt(np.mean(delta_ssh[tmask] ** 2, dtype=np.float64))
        u_rms[day] = np.sqrt(np.mean(delta_u[umask] ** 2, dtype=np.float64))
        v_rms[day] = np.sqrt(np.mean(delta_v[vmask] ** 2, dtype=np.float64))
    crossing = first_efold_crossing(lead, ssh_rms, initial_ssh_rms)
    if crossing is None:
        fit_efold = None
    else:
        fit_stop = max(2, int(np.ceil(crossing)) + 1)
        slope = np.polyfit(lead[:fit_stop], np.log(ssh_rms[:fit_stop]), 1)[0]
        fit_efold = float(1.0 / slope) if slope > 0.0 else None
    return {
        "schema": "swg-online-predictability-v1",
        "control_root": str(control_root.resolve()),
        "perturbed_root": str(perturbed_root.resolve()),
        "control_manifest_sha256": manifests[0][1],
        "perturbed_manifest_sha256": manifests[1][1],
        "source_restart_sha256": c_restart["sha256"],
        "perturb_amplitude_m": float(amplitude),
        "perturb_seed": seed,
        "lead_days": lead.tolist(),
        "ssh_rms_m": ssh_rms.tolist(),
        "u_rms_m_s": u_rms.tolist(),
        "v_rms_m_s": v_rms.tolist(),
        "first_ssh_efold_crossing_days": crossing,
        "early_log_linear_ssh_efold_days": fit_efold,
        "saturation_reached": None,
        "saturation_assessment": "not inferred from a partial deterministic twin",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--perturbed-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure_twins(args.control_root, args.perturbed_root)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    print(
        f"PASS leads={len(report['lead_days']) - 1} "
        f"ssh_efold_days={report['first_ssh_efold_crossing_days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
