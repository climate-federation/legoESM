#!/usr/bin/env python3
"""Report yearly spectral and numerical health for a no-closure SWG control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemo_testcase_l1_swg_dg_climate as climate  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

DAYS_PER_YEAR = 365


def summarize_year(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Reduce one diagnosed year while refusing non-finite hidden failures."""
    required = {
        "basin_mean_ke_m2_s2", "basin_mean_eke_m2_s2",
        "top_octave_ke_fraction", "top_octave_enstrophy_fraction",
        "native_grid_noise_index", "minimum_total_layer_thickness_m",
        "maximum_u_courant", "maximum_v_courant",
    }
    require(required <= arrays.keys(), "health diagnosis is incomplete")
    require(all(np.all(np.isfinite(arrays[name])) for name in required),
            "health diagnosis contains non-finite values")
    return {
        "basin_mean_ke_m2_s2": float(np.mean(arrays["basin_mean_ke_m2_s2"])),
        "basin_mean_eke_m2_s2": float(np.mean(arrays["basin_mean_eke_m2_s2"])),
        "top_octave_ke_fraction": float(arrays["top_octave_ke_fraction"][0]),
        "top_octave_enstrophy_fraction": float(
            arrays["top_octave_enstrophy_fraction"][0]),
        "grid_noise_index_mean": float(
            np.mean(arrays["native_grid_noise_index"])),
        "grid_noise_index_max": float(
            np.max(arrays["native_grid_noise_index"])),
        "minimum_total_layer_thickness_m": float(
            np.min(arrays["minimum_total_layer_thickness_m"])),
        "maximum_u_courant": float(np.max(arrays["maximum_u_courant"])),
        "maximum_v_courant": float(np.max(arrays["maximum_v_courant"])),
    }


def report(root: Path, deck: str) -> dict:
    """Diagnose every complete 365-day block in a content-pinned run."""
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    total_days = manifest.get("days")
    require(manifest.get("status") == "complete"
            and manifest.get("completed_days") == total_days
            and isinstance(total_days, int)
            and total_days > 0
            and total_days % DAYS_PER_YEAR == 0,
            "control run is not a complete whole-year archive")

    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card

    namelist = climate.online_control_namelist(deck)
    card = build_swg_card(namelist)
    years = []
    for year in range(1, total_days // DAYS_PER_YEAR + 1):
        start = (year - 1) * DAYS_PER_YEAR + 1
        provider = climate.legoesm_npz_provider(
            root, DAYS_PER_YEAR, start_day=start, name=f"{deck} year {year}",
            expected_namelist=namelist)
        # Top-octave, native grid-noise and thickness/Courant health exist only
        # on the explicitly selected online path (the default is registered).
        arrays = climate.diagnose(
            provider, days=DAYS_PER_YEAR, card=card, namelist=namelist,
            diagnostics="online")
        years.append({"year": year, "days": [start, start + 364],
                      **summarize_year(arrays)})
    return {
        "schema": "legoesm-swg-online-control-health-v1",
        "deck": deck,
        "diagnostics": "online",
        # Health of daily means and of snapshots differ; say which was read.
        "sampling": provider.sampling,
        "source_root": str(root.resolve()),
        "source_manifest_sha256": climate.sha256(manifest_path),
        "years": years,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deck", choices=("dg50", "dg50-nu80"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = report(args.root, args.deck)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS {len(result['years'])} yearly health rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
