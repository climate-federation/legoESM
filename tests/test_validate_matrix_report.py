"""Regression guards for the matrix validator's false-positive fixes.

Each test pins one harness false-positive that previously buried the real
signal in the cross-grid audit (2026-07-14):
  * FP1 shallow-water `p_s` range-checked against tropospheric bounds,
  * FP2 4-D (time,lat,lon,nlev) fields not masked to the active domain, so
    out-of-domain NaN fill in regional/channel cases was counted as blow-up,
  * FP3 `rest_state_*` output dirs (nested under rest_state/) not found,
  * FP4 "grew Nx" runaway fired on a machine-zero anomaly column.
"""
from __future__ import annotations

import numpy as np

from scripts.matrix import validate_matrix_report as V
from scripts.matrix.validate_matrix_report import (
    Finding,
    _range_check,
    _find_case_dir,
    _conservation_check,
)


def _land_mask(nlat=181, nlon=360, interior=slice(60, 120)):
    """A global mask that is ocean (1.0) only in an interior lat band and
    NaN elsewhere — i.e. a regional case regridded onto the global canvas."""
    lm = np.full((nlat, nlon), np.nan)
    lm[interior, :] = 1.0
    return lm


# --- FP1: shallow water has no true surface pressure ---------------------

def test_shallow_water_skips_ps_range_check():
    arrays = {"p_s": np.full((3, 181, 360), 1.25e5)}  # 1250 hPa, out of range
    fail_sw: list[Finding] = []
    _range_check(fail_sw, "atmosphere", "williamson6", "latlon", arrays,
                 V.ATMO_RANGES, eq_set="shallow_water")
    assert not any("p_s" in f.message for f in fail_sw), (
        "SW p_s (a derived rho*g*h diagnostic) must not be range-checked")
    # A hydrostatic case with the SAME field IS still checked.
    fail_hydro: list[Finding] = []
    _range_check(fail_hydro, "atmosphere", "rossby_haurwitz_6_0", "latlon",
                 arrays, V.ATMO_RANGES, eq_set="hydrostatic")
    assert any("p_s" in f.message for f in fail_hydro)


# --- FP2: mask 4-D fields to the active domain ---------------------------

def test_range_check_masks_4d_exterior_nan():
    lm = _land_mask()
    # 4-D field: finite in the interior band, NaN in the exterior fill.
    fld = np.full((2, 181, 360, 5), np.nan)
    fld[:, 60:120, :, :] = 10.0
    arrays = {"land_mask": lm, "T_3d": fld}
    findings: list[Finding] = []
    _range_check(findings, "ocean", "baroclinic_gyre", "mpas_regional",
                 arrays, V.OCEAN_RANGES)
    assert not any("NaN" in f.message for f in findings), (
        "exterior fill NaN in a 4-D field must not count as active blow-up "
        f"(got {[f.message for f in findings]})")
    # Sanity: a REAL interior NaN is still caught.
    fld[:, 60:120, :, :] = np.nan
    bad: list[Finding] = []
    _range_check(bad, "ocean", "baroclinic_gyre", "mpas_regional",
                 {"land_mask": lm, "T_3d": fld}, V.OCEAN_RANGES)
    assert any("NaN" in f.message for f in bad)


# --- FP3: locate rest_state_* output dirs --------------------------------

def test_find_case_dir_rest_state(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RESULTS_ROOT", tmp_path)
    d = tmp_path / "ocean" / "rest_state" / "rest_state_uniform_with_land" \
        / "cubed_sphere" / "C24"
    d.mkdir(parents=True)
    rec = {"test": "rest_state_uniform_with_land", "grid": "cubed_sphere",
           "resolution": "C24"}
    assert _find_case_dir("ocean", rec) == d


# --- FP4: no runaway warning on a machine-zero anomaly -------------------

def _write_cons(tmp_path, rows):
    p = tmp_path / "conservation_timeseries.csv"
    hdr = "step," + ",".join(rows[0])
    lines = [hdr] + [
        f"{i}," + ",".join(f"{r[k]:.6e}" for k in rows[0])
        for i, r in enumerate(rows)
    ]
    p.write_text("\n".join(lines))
    return tmp_path


def test_runaway_skips_machine_zero_anomaly(tmp_path):
    # A conserved anomaly ~0: 1.6e-18 -> 8.9e-16 is "500x" but physically zero.
    d = _write_cons(tmp_path, [{"volume": 1.6e-18}, {"volume": 8.9e-16}])
    findings: list[Finding] = []
    _conservation_check(findings, "ocean", "phillips_two_layer", "cubed_sphere", d)
    assert not any("grew" in f.message for f in findings)
    # A genuinely runaway, well-scaled series IS still flagged.
    d2 = _write_cons(tmp_path, [{"volume": 1.0e15}, {"volume": 1.0e17}])
    real: list[Finding] = []
    _conservation_check(real, "ocean", "blowup", "latlon", d2)
    assert any("grew" in f.message for f in real)
    # And a small-but-real baseline that blows up (1e-12 -> 1e-4, series_max
    # clears the floor) must NOT be suppressed — guards the over-tight
    # abs(v0) / same-sign clauses that a first fix wrongly added.
    d3 = _write_cons(tmp_path, [{"volume": 1.0e-12}, {"volume": 1.0e-4}])
    near0: list[Finding] = []
    _conservation_check(near0, "ocean", "slow_blowup", "latlon", d3)
    assert any("grew" in f.message for f in near0)


def test_select_active_square_grid_axis():
    """A square active mask (nlat==nlon) plus a field whose LEADING axis also
    equals nlat must still mask over (lat, lon), not (time, lat)."""
    n = 8
    lm = np.full((n, n), np.nan)
    lm[3:6, :] = 1.0                       # interior ocean band in lat
    fld = np.full((n, n, n, 2), 5.0)       # (time=n, lat=n, lon=n, nlev=2)
    fld[:, 3:6, :, :] = np.nan             # real NaN inside the active band
    findings: list[Finding] = []
    _range_check(findings, "ocean", "sq", "cubed_sphere",
                 {"land_mask": lm, "T_3d": fld}, V.OCEAN_RANGES)
    assert any("NaN" in f.message for f in findings), (
        "interior NaN over the (lat,lon) active band must be caught even when "
        "a leading time axis shares the grid size")


def test_non_2d_mask_is_graceful():
    """An unstructured (time, ncells) land_mask must not crash the validator;
    it is simply left unapplied (no active mask)."""
    lm = np.ones((3, 100))                 # (time, ncells) — not (nlat, nlon)
    arrays = {"land_mask": lm, "eta": np.zeros((3, 100))}
    findings: list[Finding] = []
    _range_check(findings, "ocean", "unstructured", "mpas", arrays,
                 V.OCEAN_RANGES)  # must not raise
