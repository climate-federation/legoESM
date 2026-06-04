"""One-off driver: Eady GM/Redi with strong κ and β_S=0 for clear visual signal.

Runs the 6 cases (3 modes × 2 grids) with:
  - kappa_GM_override = kappa_Redi_override = 1e4   (10× the matrix default)
  - beta_S_override = 0.0                            (T = f(ρ) exactly)
  - 90-day integration                                (default matrix is 30 d)

Output goes under `results/ocean/eady_gm_redi_strong/<case>/<grid>/<res>/`
so it does not collide with the regular matrix layout.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run_eady_gm_redi_strong.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Ensure the repo's scripts directory is on sys.path so we can import
# the test-matrix experiment modules without installing them.
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocean_test_matrix.experiments import run_eady_gm_redi  # noqa: E402
from ocean_test_matrix.testcase import TestCase             # noqa: E402


DAYS = 90.0
KAPPA = 1.0e4
OUT_ROOT = REPO_ROOT / "results" / "ocean" / "eady_gm_redi_strong"


def _build_case(case_name: str, grid_type: str, resolution: str,
                gm_mode: str) -> TestCase:
    return TestCase(
        case=case_name,
        grid_type=grid_type,
        resolution=resolution,
        duration_days=DAYS,
        quick_days=DAYS,
        run_kwargs={
            "gm_mode": gm_mode,
            "slope_scheme": "centered",
            "kappa_GM_override": KAPPA,
            "kappa_Redi_override": KAPPA,
            "beta_S_override": 0.0,
        },
    )


CASES = [
    # latlon_channel
    ("eady_gm_redi_gm_only",   "latlon_channel", "20x10",  "gm_only"),
    ("eady_gm_redi_redi_only", "latlon_channel", "20x10",  "redi_only"),
    ("eady_gm_redi",           "latlon_channel", "20x10",  "gm_redi"),
    # mpas_channel
    ("eady_gm_redi_gm_only_mpas",   "mpas_channel", "100km", "gm_only"),
    ("eady_gm_redi_redi_only_mpas", "mpas_channel", "100km", "redi_only"),
    ("eady_gm_redi_mpas",           "mpas_channel", "100km", "gm_redi"),
]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUT_ROOT}")
    print(f"days={DAYS}  kappa_GM=kappa_Redi={KAPPA}  beta_S=0.0\n")

    summary: list[tuple[str, str, float, str]] = []
    for i, (case_name, grid_type, res, gm_mode) in enumerate(CASES, 1):
        tc = _build_case(case_name, grid_type, res, gm_mode)
        out_dir = OUT_ROOT / case_name / grid_type / res
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(CASES)}] {case_name}/{grid_type}/{res} "
              f"({DAYS}d, mode={gm_mode})")
        t0 = time.time()
        status, _max_speed, notes = run_eady_gm_redi(tc, out_dir, DAYS)
        dt = time.time() - t0
        print(f"   {status} | {dt:6.1f}s | {notes}\n")
        summary.append((case_name, status, dt, notes))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for case_name, status, dt, notes in summary:
        print(f"  {status:6s} | {dt:6.1f}s | {case_name:40s} | {notes}")


if __name__ == "__main__":
    main()
