"""Issue 506 evidence harness: Case-6 (Rossby-Haurwitz wave-4) long-run symmetry.

Runs the SAME ``run_shallow_water`` W6 path (no duplicated numerics) on the cube
at C48 and C96 out to extended duration, saving height snapshots.  Then measures
the longitudinal 4-fold symmetry purity of the day-N height field:

    sym_err(t) = || h - rot90(h_zonal_structure) ||  is hard on the cube, so we
    use the cube-robust metric: project each snapshot onto its wavenumber-4
    longitudinal Fourier component (after regridding to lat-lon) and report the
    fraction of off-m=4 power that grows as the wave breaks down.

The runner already regrids + saves; here we just drive the integrations and
report L-infinity height extremes + wall time.  Symmetry analysis reads the
saved snapshot npz post-hoc (see analyze() ).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_MATRIX = _ROOT / "scripts" / "matrix" / "run_atmosphere_test_matrix.py"
_spec = importlib.util.spec_from_file_location("_atm_matrix", _MATRIX)
m = importlib.util.module_from_spec(_spec)
sys.modules["_atm_matrix"] = m
_spec.loader.exec_module(m)

OUT = _ROOT / "results" / "diag_506_case6"
OUT.mkdir(parents=True, exist_ok=True)

# (label, resolution, days) — issue: breaks ~day20 at C48; C96 reportedly worse.
runs = [
    ("C48_28d", "C48", 28.0),
    ("C96_20d", "C96", 20.0),
]

print("# Case-6 long-run symmetry  (cube)")
print(f"{'label':10s} {'res':5s} {'days':>5s}  {'status':6s}  notes")
for label, res, days in runs:
    tc = m.TestCase(
        equation_set="shallow_water", case="williamson6",
        grid_type="cubed_sphere", resolution=res, vertical_coord="none",
        duration_days=days, quick_days=1.0, run_kwargs={"test_num": 6},
    )
    odir = OUT / label
    odir.mkdir(parents=True, exist_ok=True)
    status, wall, notes = m.run_shallow_water(tc, odir, days)
    print(f"{label:10s} {res:5s} {days:5.0f}  {status:6s}  {notes}  [{wall:.0f}s] -> {odir}")

print("\n# DONE. height/wind_speed PNGs + snapshots npz under each label dir.")
