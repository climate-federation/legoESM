#!/usr/bin/env python3
"""Report yearly dg50-restricted truth stationarity from listed fine days."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DAYS_PER_YEAR = 365


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_velocity(fields: dict[str, np.ndarray], masks: dict) -> tuple[np.ndarray, np.ndarray]:
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import velocity_to_cell_centre

    u_native = np.where(masks["U"], fields["U"], 0.0)
    v_native = np.where(masks["V"], fields["V"], 0.0)
    u = np.concatenate((np.zeros_like(u_native[:, :1]), u_native), axis=1)
    v = np.concatenate((np.zeros_like(v_native[:1, :]), v_native), axis=0)
    u_center, v_center = velocity_to_cell_centre(jnp.asarray(u), jnp.asarray(v))
    return np.asarray(u_center), np.asarray(v_center)


def yearly_metrics(frames: list[dict[str, np.ndarray]], geometry: dict) -> dict:
    """Return KE, EKE, and mean velocity for one pre-grouped year."""
    if not frames:
        raise ValueError("stationarity year has no frames")
    masks = geometry["mask"]
    wet = np.asarray(masks["T"], dtype=bool)
    area = np.where(wet, np.asarray(geometry["area"]["T"]), 0.0)
    area_sum = np.sum(area, dtype=np.float64)
    velocities = [_cell_velocity(frame, masks) for frame in frames]
    mean_u = np.mean([item[0] for item in velocities], axis=0, dtype=np.float64)
    mean_v = np.mean([item[1] for item in velocities], axis=0, dtype=np.float64)
    ke = np.asarray([
        np.sum(0.5 * (u * u + v * v) * area, dtype=np.float64) / area_sum
        for u, v in velocities
    ])
    eke = np.asarray([
        np.sum(0.5 * ((u - mean_u) ** 2 + (v - mean_v) ** 2) * area,
               dtype=np.float64) / area_sum
        for u, v in velocities
    ])
    return {
        "mean_u": mean_u,
        "mean_v": mean_v,
        "mean_ke_m2_s-2": float(np.mean(ke, dtype=np.float64)),
        "mean_eke_m2_s-2": float(np.mean(eke, dtype=np.float64)),
        "ke_first_half_m2_s-2": float(np.mean(
            ke[:max(1, len(ke) // 2)], dtype=np.float64)),
        "ke_second_half_m2_s-2": float(np.mean(
            ke[len(ke) // 2:], dtype=np.float64)),
    }


def scan_truth(root: Path, *, sampling: str) -> dict:
    """Restrict complete manifest-listed years and report yearly stationarity."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
    )
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        conservative_restrict_cgrid,
    )

    if sampling not in ("snapshot", "daily_mean"):
        raise ValueError("sampling must be snapshot or daily_mean")
    manifest_path = (root / "manifest.json" if (root / "manifest.json").is_file()
                     else root / "manifest.pending.json")
    manifest = json.loads(manifest_path.read_text())
    pins = manifest.get("daily_sha256", {})
    completed_days = int(manifest.get("completed_days", 0))
    complete_years = completed_days // DAYS_PER_YEAR
    if complete_years < 1:
        raise ValueError("truth manifest has no complete 365-day year")
    field_keys = ({"T": "ssh_m", "U": "u_m_s", "V": "v_m_s"}
                  if sampling == "snapshot" else {
                      "T": "ssh_daily_mean_m",
                      "U": "u_daily_mean_m_s",
                      "V": "v_daily_mean_m_s",
                  })
    source = cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = cgrid_geometry(SWG_DG50_NAMELIST)
    years = []
    previous = None
    admitted_hashes = []
    for year in range(1, complete_years + 1):
        frames = []
        first = (year - 1) * DAYS_PER_YEAR + 1
        last = year * DAYS_PER_YEAR
        for day in range(first, last + 1):
            path = root / f"day{day:04d}.npz"
            expected = pins.get(path.name)
            actual = sha256(path)
            if expected != actual:
                raise ValueError(f"unlisted or changed truth day: {path.name}")
            admitted_hashes.append(actual)
            with np.load(path, allow_pickle=False) as saved:
                missing = set(field_keys.values()) - set(saved.files)
                if missing:
                    raise ValueError(f"{path.name}: missing {sorted(missing)}")
                fine = {kind: np.asarray(saved[key], dtype=np.float64)
                        for kind, key in field_keys.items()}
            frames.append(conservative_restrict_cgrid(fine, source, target))
        metrics = yearly_metrics(frames, target)
        wet = np.asarray(target["mask"]["T"], dtype=bool)
        area = np.where(wet, np.asarray(target["area"]["T"]), 0.0)
        if previous is None:
            velocity_difference = None
        else:
            delta2 = ((metrics["mean_u"] - previous["mean_u"]) ** 2
                      + (metrics["mean_v"] - previous["mean_v"]) ** 2)
            velocity_difference = float(np.sqrt(
                np.sum(delta2 * area, dtype=np.float64)
                / np.sum(area, dtype=np.float64)))
        years.append({
            "year": year,
            "days": [first, last],
            "mean_ke_m2_s-2": metrics["mean_ke_m2_s-2"],
            "mean_eke_m2_s-2": metrics["mean_eke_m2_s-2"],
            "ke_first_half_m2_s-2": metrics["ke_first_half_m2_s-2"],
            "ke_second_half_m2_s-2": metrics["ke_second_half_m2_s-2"],
            "mean_velocity_difference_from_previous_year_m_s": (
                velocity_difference),
        })
        previous = metrics
    aggregate = hashlib.sha256("".join(admitted_hashes).encode()).hexdigest()
    return {
        "schema": "swg-online-reference-stationarity-v1",
        "source_root": str(root.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_status": manifest.get("status"),
        "completed_days_at_scan": completed_days,
        "sampling": sampling,
        "complete_years": complete_years,
        "admitted_daily_digest_aggregate_sha256": aggregate,
        "years": years,
        "stationarity_verdict": "UNMEASURED: no acceptance bar is registered",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--sampling", choices=("snapshot", "daily_mean"),
                        default="daily_mean")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = scan_truth(args.truth_root, sampling=args.sampling)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    print(f"PASS complete_years={report['complete_years']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
