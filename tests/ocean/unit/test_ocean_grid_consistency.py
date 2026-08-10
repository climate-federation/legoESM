"""Direct test for scripts/validate/ocean_grid_consistency.py.

The decisive one is ``test_land_fill_value_does_not_move_the_metric``:
arms disagree about what a LAND cell holds (lat-lon pins 0 degC, MPAS
Neumann-fills ~20 degC), so if the metric moves when the land fill
changes, it is measuring the land convention rather than the dycores --
which is exactly what an earlier revision did.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load():
    p = _REPO / "scripts" / "validate" / "ocean_grid_consistency.py"
    spec = importlib.util.spec_from_file_location("_ocn_cons", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ocn_cons"] = mod
    spec.loader.exec_module(mod)
    return mod


def _arm(root: Path, case: str, grid: str, res: str, field_vals,
         land_fill: float, nlat=10, nlon=12):
    """One arm whose LAND cells hold *land_fill*."""
    d = root / case / case / grid / res
    d.mkdir(parents=True)
    mask = np.ones((nlat, nlon))
    mask[:2, :] = 0.0                      # polar land, as in the suite
    mask[-2:, :] = 0.0
    sst = np.where(mask > 0.5, field_vals, land_fill)
    np.savez(d / "snapshots_latlon.npz",
             times_days=np.array([0.0, 1.0]),
             lat=np.linspace(-85, 85, nlat), lon=np.linspace(0, 330, nlon),
             SST=np.stack([sst, sst]), eta=np.zeros((2, nlat, nlon)),
             u_sfc=np.zeros((2, nlat, nlon)),
             land_mask=np.stack([mask, mask]))
    (d / "results.txt").write_text("status: PASS\n")


def test_erode_does_not_wrap_across_the_poles():
    """Latitude is NOT periodic: eroding must not pull the south pole's
    neighbour into the north pole's stencil."""
    mod = _load()
    m = np.ones((5, 6), dtype=bool)
    m[0, :] = False                     # land along the SOUTH edge only
    out = mod._erode(m)
    assert not out[1].any(), "row adjacent to south land must be eroded"
    assert out[3].all(), "north rows must be untouched by south land"
    # Longitude DOES wrap: land at column 0 must erode the last column.
    m2 = np.ones((5, 6), dtype=bool)
    m2[:, 0] = False
    out2 = mod._erode(m2)
    assert not out2[2, -1], "longitude must wrap when eroding"


def test_land_fill_value_does_not_move_the_metric(tmp_path):
    rng = np.random.default_rng(0)
    ocean = 10.0 + rng.random((10, 12))
    mod = _load()
    results = []
    for fill_b in (0.0, 20.5):          # lat-lon convention vs MPAS-like
        root = tmp_path / f"fill{fill_b}"
        _arm(root, "geostrophic_adjustment", "latlon", "36x72", ocean, 0.0)
        _arm(root, "geostrophic_adjustment", "mpas", "ico4", ocean, fill_b)
        pairs, _ref = mod.cross_grid_rms(
            root, "geostrophic_adjustment", ["latlon", "mpas"])
        results.append(pairs["latlon|mpas"]["rms_abs"])
    assert results[0] == pytest.approx(results[1], abs=1e-12), (
        f"metric moved when only the LAND fill changed: {results} -- it is "
        f"measuring the land convention, not the dycores")
    assert results[0] < 1e-12, "identical ocean fields must compare equal"


def test_stale_artifact_is_refused(tmp_path, capsys):
    """An artifact at a resolution the matrix no longer registers must be
    ignored loudly, not silently mixed into the comparison."""
    mod = _load()
    ocean = np.full((10, 12), 12.0)
    _arm(tmp_path, "geostrophic_adjustment", "mpas", "ico3", ocean, 0.0)
    assert mod._find(tmp_path, "geostrophic_adjustment", "mpas") is None
    assert "stale artifact ignored" in capsys.readouterr().out


def test_registered_resolution_honours_per_case_overrides():
    """barotropic_wave overrides the per-grid default for latlon."""
    mod = _load()
    assert mod._registered_resolution("barotropic_wave", "latlon") == "48x72"
    assert mod._registered_resolution("geostrophic_adjustment",
                                      "latlon") == "36x72"
