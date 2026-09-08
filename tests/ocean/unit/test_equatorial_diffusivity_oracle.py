"""The equatorial diffusivity probe must reduce both models the same way.

The probe compares NEMO's published ``avm``/``avt`` against ours in the cold
tongue. Two things can silently make that comparison meaningless: reducing the
two sides differently, and accepting a snapshot that never stored the
diffusivities at all. Both are pinned here, and the depth-axis convention --
interfaces, not cell centres -- is pinned with a case whose answer is known by
construction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_eq_diff_oracle",
    REPO / "scripts" / "validate" / "ocean_fidelity"
    / "equatorial_diffusivity_oracle.py")


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    sys.modules["_eq_diff_oracle"] = mod
    _SPEC.loader.exec_module(mod)
    return mod


def _snapshot(tmp_path, name, *, with_k=True, with_zint=True, land_col=None,
              nk=4):
    """A minimal snapshot. The closure's K live at the nlev-1=4 interior
    interfaces; z_interface_ref carries their true depths (as the driver now
    stores), so the probe never reconstructs a cell-centre midpoint."""
    ny, nx, nlev = 3, 4, 5
    zc = np.array([1.0, 3.0, 7.0, 15.0, 31.0])
    z_int = np.array([2.0, 5.0, 11.0, 23.0])            # true interior faces
    lat = np.tile(np.array([[-1.0], [0.0], [1.0]]), (1, nx))
    lon = np.tile(np.linspace(210.0, 250.0, nx), (ny, 1))
    mask = np.ones((ny, nx))
    if land_col is not None:
        mask[:, land_col] = 0.0
    kw = dict(lat_T=lat, lon_T=lon, land_mask=mask, z_center_ref=zc,
              T=np.zeros((ny, nx, nlev)))
    if with_zint:
        kw["z_interface_ref"] = z_int
    if with_k:
        kw["K_M_diag"] = np.full((ny, nx, nk), 4.0e-3)
        kw["K_H_diag"] = np.full((ny, nx, nk), 1.0e-3)
    p = tmp_path / name
    np.savez(p, **kw)
    return p


def test_a_snapshot_without_the_diffusivities_is_refused(tmp_path):
    """A missing measurement must be an error, not a fallback -- an inferred
    profile compared against NEMO's published one is a different quantity."""
    mod = _load()
    p = _snapshot(tmp_path, "no_k.npz", with_k=False)
    with pytest.raises(SystemExit, match="kprofile-snapshots"):
        mod._load_ours(p)


def test_the_true_interface_depths_are_used_not_a_midpoint(tmp_path):
    """K lives at interfaces, and the snapshot now carries their true depths.
    A cell-centre midpoint reconstruction is a DIFFERENT axis -- on ORCA1 it
    shifted a 64.96 m interface to 65.12 m and dropped it from a <=65 m window
    NEMO's 64.98 m depthw kept -- so the probe must read z_interface_ref."""
    mod = _load()
    avm, avt, bn2, lat, lon, zk = mod._load_ours(
        _snapshot(tmp_path, "iface.npz"))
    assert bn2 is None
    assert np.allclose(zk, [2.0, 5.0, 11.0, 23.0])       # the stored faces
    assert avm.shape[-1] == zk.size


def test_a_snapshot_without_interface_depths_is_refused_not_guessed(tmp_path):
    """An old snapshot lacking z_interface_ref must not fall back to a
    midpoint guess for a diffusivity's depth axis."""
    mod = _load()
    with pytest.raises(SystemExit, match="z_interface_ref"):
        mod._load_ours(_snapshot(tmp_path, "old.npz", with_zint=False))


def test_a_depth_axis_that_disagrees_with_the_data_fails(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit, match="disagree"):
        mod._load_ours(_snapshot(tmp_path, "bad.npz", nk=3))


def test_land_columns_are_excluded_from_the_reduction(tmp_path):
    """A dry column carrying a zero or a floor would drag the median toward
    it, and the land fraction differs between our mesh and NEMO's."""
    mod = _load()
    avm, avt, _, lat, lon, zk = mod._load_ours(
        _snapshot(tmp_path, "land.npz", land_col=0))
    assert np.isnan(avm[:, 0, :]).all() and np.isnan(avt[:, 0, :]).all()
    assert np.isfinite(avm[:, 1:, :]).all()


def test_the_table_reduces_the_ratio_per_column_not_the_ratio_of_medians(
        capsys):
    """Prandtl is a ratio taken per column, then reduced. A ratio of
    reductions is a different number wherever the two fields are not
    proportional, which is the case that matters."""
    mod = _load()
    ny, nx, nk = 2, 2, 3
    z = np.array([10.0, 30.0, 50.0])
    avm = np.zeros((ny, nx, nk))
    avt = np.zeros((ny, nx, nk))
    # Column-wise Pr is [20, 2, 2, 2], whose median is 2. The ratio of the
    # medians is median(avm)/median(avt) = 3e-3/1e-3 = 3 -- a different
    # number, and the one an over-viscous outlier column would produce.
    avm[..., :] = 2.0e-3
    avt[..., :] = 1.0e-3
    avm[0, 0, :] = 20.0e-3
    avt[0, 0, :] = 1.0e-3
    avm[0, 1, :] = 4.0e-3
    avt[0, 1, :] = 2.0e-3
    band = np.ones((ny, nx), dtype=bool)
    summary = mod._table(avm, avt, None, band, z, 300.0, "unit")
    out = capsys.readouterr().out
    assert summary["pr"] == pytest.approx(2.0)
    assert summary["avt"] == pytest.approx(1.0e-3)
    assert "unit" in out and "SURFACE closure" in out
    # avm is reported separately so a ratio defect (avm~NEMO, avt<<NEMO) is
    # distinguishable from starvation (both small) -- GLM 2026-08-23.
    assert summary["sfc_avm"] == pytest.approx(3.0e-3)   # median of avm itself
    # The ratio of the medians would have been 3.0; reporting that would let
    # a single over-viscous column set the number for the whole box.
    assert summary["pr"] != pytest.approx(3.0)


def test_surface_and_entrainment_bands_are_summarised_separately():
    """The cooling budget is generated in the 65-105 m entrainment zone, not
    the top 60 m, so the two bands must be reported apart -- averaging them
    reports a background ratio as the closure's (GLM 2026-08-23)."""
    mod = _load()
    z = np.array([10.0, 40.0, 80.0, 100.0])
    ny, nx = 1, 1
    avm = np.array([[[3e-3, 3e-3, 6e-6, 6e-6]]])
    avt = np.array([[[1e-3, 1e-3, 5e-6, 5e-6]]])
    s = mod._table(avm, avt, None, np.ones((ny, nx), bool), z, 300.0, "b")
    assert "sfc_avm" in s and "ent_avm" in s
    assert s["sfc_avt"] == pytest.approx(1e-3)     # surface closure value
    assert s["ent_avt"] == pytest.approx(5e-6)     # entrainment (deeper) value
    assert s["sfc_avt"] != s["ent_avt"]


def test_a_box_with_no_finite_diffusivities_fails_loudly():
    mod = _load()
    z = np.array([10.0, 30.0])
    nan = np.full((2, 2, 2), np.nan)
    with pytest.raises(SystemExit, match="no finite diffusivities"):
        mod._table(nan, nan, None, np.ones((2, 2), dtype=bool), z, 300.0, "x")
