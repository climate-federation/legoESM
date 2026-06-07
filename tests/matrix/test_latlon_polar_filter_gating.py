"""Pin the lat-lon polar-filter / dt gating in the atmosphere matrix runner.

The runner enables the CAM-FV polar Fourier filter (and the cube-like dt=200)
for lat-lon hydrostatic PE cases, EXCEPT the ``rotated_*`` (DCMIP-2008) cases
whose tilted solid-body jet flows across the grid poles — there the zonal
filter would damp the physical cross-polar flow, so it must stay OFF with the
legacy pole-limited dt.  Codex review asked for a test pinning the exact
exception set, the env override, and dt/flag agreement.
"""
import importlib
import sys
from pathlib import Path

import pytest

_MATRIX_DIR = Path(__file__).resolve().parents[2] / "scripts" / "matrix"


@pytest.fixture(scope="module")
def mat():
    sys.path.insert(0, str(_MATRIX_DIR))
    try:
        m = importlib.import_module("run_atmosphere_test_matrix")
    finally:
        # leave it importable for other tests; cheap idempotent insert
        pass
    return m


# Every lat-lon hydrostatic case the runner builds, and whether the polar
# filter must be ON.  Cross-polar-flow (rotated) cases -> OFF.
_LATLON_HYDRO_CASES = {
    "held_suarez": True,
    "held_suarez_topo": True,
    "baroclinic": True,
    "rotated_baroclinic": False,
    "rotated_steady": False,
    "rest_state_topo": True,
    "gravity_wave_3_1": True,
    "inertio_gravity_3_2": True,
    "mountain_rossby_5_0": True,
    "rossby_haurwitz_6_0": True,
    "amip": True,
}


def test_only_rotated_cases_disable_filter(mat, monkeypatch):
    monkeypatch.delenv("LEGOESM_LATLON_POLAR_FILTER", raising=False)
    for case, expected_on in _LATLON_HYDRO_CASES.items():
        assert mat._latlon_polar_filter_on(case) is expected_on, (
            f"{case}: filter_on={mat._latlon_polar_filter_on(case)}, "
            f"expected {expected_on}")


def test_rotated_set_matches_case_families(mat, monkeypatch):
    """The string gate ``'rotated' in case`` catches exactly the case names
    that legoESM tags as rotated (no over/under-match)."""
    monkeypatch.delenv("LEGOESM_LATLON_POLAR_FILTER", raising=False)
    rotated = {c for c in mat._CASE_FAMILIES if "rotated" in c}
    assert rotated == {"rotated_baroclinic", "rotated_steady"}
    for c in mat._CASE_FAMILIES:
        # Only the rotated names disable the filter via the substring rule.
        disabled = not mat._latlon_polar_filter_on(c)
        assert disabled == (c in rotated) or "rotated" not in c


def test_env_override_forces_filter_off(mat, monkeypatch):
    monkeypatch.setenv("LEGOESM_LATLON_POLAR_FILTER", "0")
    for case in _LATLON_HYDRO_CASES:
        assert mat._latlon_polar_filter_on(case) is False


def test_dt_agrees_with_filter_flag(mat, monkeypatch):
    """dt == dt_cap exactly when the filter is on; else pole-limited (<dt_cap)."""
    monkeypatch.delenv("LEGOESM_LATLON_POLAR_FILTER", raising=False)
    dx_pole = 6.0e3  # ~ 72x144 polar cell; pole-limited dt = 0.5*6e3/300 = 10 s
    for case, expected_on in _LATLON_HYDRO_CASES.items():
        dt = mat._latlon_dt(dx_pole, 200.0, case)
        if expected_on:
            assert dt == 200.0, f"{case}: filter on but dt={dt}"
        else:
            assert dt < 200.0, f"{case}: filter off but dt={dt} not pole-limited"
            assert dt == pytest.approx(0.5 * dx_pole / 300.0)
