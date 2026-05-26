"""Cross-grid RCE smoke at the iter-12 production-validated configs.

iter-12 (2026-05) measured 30-day RCE PASS on all 4 hydrostatic grid
families. These tests run a SHORTER (2-day) version of the same
configs as a CI-friendly regression so future commits don't quietly
break any of the grid×discretization paths.

Wall-time budget: 4 grids × ~10 s wall at 2 days = ~40 s total.

For the FULL 30-day production validation see
``scripts/run_rce_cross_grid.sh`` and CRM_implementation.md iter-12.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_RCE = REPO_ROOT / "scripts" / "run_rce.py"


def _run_rce(grid_type, discretization, resolution, days, output_dir):
    """Invoke scripts/run_rce.py with the iter-12-validated CLI."""
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(RUN_RCE),
        "--grid-type", grid_type,
        "--discretization", discretization,
        "--resolution", str(resolution),
        "--days", str(days),
        "--diag-days", "1",
        "--nlev", "20",
        "--output", str(output_dir),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600,
    )
    return result


def _parse_results(output_dir):
    """Read ``results.txt`` and return a dict of fields."""
    results_txt = output_dir / "results.txt"
    if not results_txt.exists():
        return None
    fields = {}
    for line in results_txt.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


def _parse_notes(notes: str) -> dict:
    """Parse `mean_T_sfc=..., mean_T=..., max|v|=...` into a dict."""
    out = {}
    for part in notes.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            pass
    return out


# Coverage matrix (iter-13). Includes C48 deliberately so any future
# regression of the auto-dt ladder (e.g. accidentally restoring
# dt=300 for N>24) trips the BLOWUP gate at 2 days and FAILS this
# test. iter-13 measurements: C48 at dt=150 (auto) reached only
# max|v|≈1.7 m/s by day 2, well inside the production envelope.
@pytest.mark.parametrize("grid_type,discretization,resolution", [
    ("cubed_sphere", "cdgrid", 12),
    ("cubed_sphere", "cdgrid", 48),   # iter-13 auto-dt=150 stability
    ("latlon", "latlon_cgrid", 16),
    ("voronoi", "mpas", 4),
    ("gaussian", "spectral", 21),
])
def test_rce_2day_smoke_passes(tmp_path, grid_type, discretization, resolution):
    """Per-grid 2-day RCE smoke must:
    1. exit cleanly (no crashes, no NaN).
    2. report status == PASS in results.txt.
    3. mean_T_sfc within 1 K of the IC (300 K).
    4. max|v| < 50 m/s (well under the 200 m/s BLOWUP gate; iter-12
       production runs landed at 2-12 m/s; >50 m/s in a 2-day smoke
       is a smoking gun for an in-flight CFL crash).

    These bounds reflect the iter-12 / iter-13 30-day production
    measurements where all grids land at mean_T_sfc within ±1 K of IC
    and max|v| ≤ 12 m/s at equilibrium.

    Coverage includes C48 (iter-13 auto-dt=150) so any future
    regression of the auto-dt ladder (e.g. accidentally widening
    the dt=600 branch back to N≤48) trips this test by day 2.
    """
    out_dir = tmp_path / f"{grid_type}_{resolution}"
    result = _run_rce(grid_type, discretization, resolution, days=2,
                      output_dir=out_dir)
    if result.returncode != 0:
        pytest.fail(
            f"{grid_type}/{discretization}@{resolution} exited nonzero "
            f"({result.returncode})\nstdout tail:\n{result.stdout[-2000:]}"
            f"\nstderr tail:\n{result.stderr[-1000:]}"
        )
    fields = _parse_results(out_dir)
    assert fields is not None, (
        f"{grid_type} did not produce results.txt — script failed silently?"
    )
    assert fields.get("status") == "PASS", (
        f"{grid_type}/{discretization}@{resolution}: "
        f"status={fields.get('status')} (want PASS). "
        f"notes={fields.get('notes')}"
    )
    notes_dict = _parse_notes(fields.get("notes", ""))
    t_sfc = notes_dict.get("mean_T_sfc")
    assert t_sfc is not None, (
        f"{grid_type}: results.txt notes line missing mean_T_sfc — "
        f"unexpected format: {fields.get('notes')!r}"
    )
    assert abs(t_sfc - 300.0) < 1.0, (
        f"{grid_type}/{discretization}@{resolution}: "
        f"mean_T_sfc={t_sfc:.2f} drifted >1 K from IC 300 K — "
        f"slab-ocean coupling or radiation regression suspected. "
        f"See CRM_implementation.md iter-12 for expected values."
    )
    max_v = notes_dict.get("max|v|")
    assert max_v is not None, (
        f"{grid_type}: notes line missing max|v| — see iter-13 PASS "
        f"diagnostics emitter in scripts/run_rce.py."
    )
    assert max_v < 50.0, (
        f"{grid_type}/{discretization}@{resolution}: "
        f"max|v|={max_v:.2f} m/s at 2-day exceeds 50 m/s sanity cap. "
        f"Production envelope is 2-12 m/s (iter-12/iter-13 measured); "
        f">50 m/s in a 2-day smoke means a CFL crash in flight even "
        f"though the BLOWUP gate at 200 m/s has not fired yet. "
        f"Bisect against the iter-13 auto-dt ladder + CRM_implementation.md."
    )
