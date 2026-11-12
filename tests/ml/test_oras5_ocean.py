"""Direct tests of ``legoesm.training.oras5_ocean`` and ``scripts/run/train_ocean_wb_physics.py``
helpers (no data, no model): layouts, masks, face averaging, loss zero/scale, level guard."""
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training import oras5_ocean as O

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.run import train_ocean_wb_physics as T  # noqa: E402


def _day():
    a = np.zeros((4 * O.NLEV + 4, O.NLAT, O.NLON), np.float32)
    a[:O.NLEV] = 10.0                                     # T
    a[O.NLEV:2 * O.NLEV] = 35.0                           # S
    a[2 * O.NLEV:3 * O.NLEV] = np.arange(O.NLON)[None, None, :]   # u = column index
    a[3 * O.NLEV:4 * O.NLEV] = np.arange(O.NLAT)[None, :, None]   # v = row index
    a[4 * O.NLEV] = 0.3
    a[:, 100, 50] = np.nan                                # an ORAS5 gap in a wet cell
    return a


def test_day_to_fields_layout_faces_and_gap():
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = np.ones((O.NLAT, O.NLON + 1)), np.ones((O.NLAT + 1, O.NLON))
    f, v = O.day_to_fields(_day(), wet3, um, vm)
    assert f["T"].shape == (180, 360, 75) and f["u"].shape == (180, 361, 75)
    assert f["v"].shape == (181, 360, 75) and f["eta"].shape == (180, 360)
    assert np.isclose(f["T"][100, 50, 3], 10.0) and not v["T"][100, 50, 3]   # filled, masked
    assert np.isclose(f["u"][10, 5, 0], 4.5) and np.isclose(f["u"][10, 0, 0], 179.5)
    assert np.isclose(f["u"][10, 360, 0], 179.5) and np.isclose(f["v"][10, 7, 0], 9.5)
    assert not v["v"][0].any() and not v["v"][-1].any()      # polar faces carry no target


def test_loss_zero_at_truth_and_std_scaled():
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = np.ones((O.NLAT, O.NLON + 1)), np.ones((O.NLAT + 1, O.NLON))
    f, v = O.day_to_fields(_day(), wet3, um, vm)
    w = O.weights(O.base_weights(np.arange(-89.5, 90.0), np.ones(O.NLEV)), v)
    std = {k: 2.0 for k in O.FIELDS}
    t = {k: jnp.asarray(f[k]) for k in O.FIELDS}
    assert float(O.l1_loss(t, t, w, std)) == 0.0
    p = dict(t, T=t["T"] + 1.0)                     # +1 K everywhere -> 1/std/5
    assert np.isclose(float(O.l1_loss(p, t, w, std)), 0.5 / 5)
    p = dict(t, eta=t["eta"] + 1.0)                 # uniform eta offset is removed
    assert np.isclose(float(O.l1_loss(p, t, w, std)), 0.0)


def test_check_levels(tmp_path):
    d = np.linspace(0.5, 5900.0, O.NLEV)
    (tmp_path / "meta.json").write_text(json.dumps({"depths_m": d.tolist()}))
    O.check_levels(tmp_path, -(d + 0.6))
    with pytest.raises(ValueError, match="differ"):
        O.check_levels(tmp_path, d + 1.5)


def test_sample_dates_in_range_and_reproducible():
    a = T.sample_dates(1993, 2017, 2, 4, 0)
    assert a == T.sample_dates(1993, 2017, 2, 4, 0) and len(set(a)) == 4
    assert all(np.datetime64("1993-01-01") <= d <= np.datetime64("2017-12-29") for d in a)


def test_forcing_must_be_named():
    with pytest.raises(SystemExit):
        T.parse_args(["--store", "s", "--out", "o"])
    assert T.parse_args(["--store", "s", "--out", "o", "--forcing", "none"]).days == 2


def test_loss_weights_area_and_thickness_unequally():
    """A localised error is scaled by its cell's area x thickness share, not counted uniformly."""
    lat = np.arange(-89.5, 90.0)
    h = np.ones((O.NLAT, O.NLON, O.NLEV))
    h[..., 1] = 3.0                                          # level 1 three times thicker
    base = O.base_weights(lat, h)
    v = {k: np.ones(base[k].shape, bool) for k in O.FIELDS}
    w = O.weights(base, v)
    t = {k: jnp.zeros(base[k].shape) for k in O.FIELDS}
    std = {k: 1.0 for k in O.FIELDS}
    p = dict(t, T=t["T"].at[90, 0, 1].set(1.0))              # one equatorial cell, thick level
    wt = np.cos(np.deg2rad(lat))[:, None, None] * h
    expect = np.cos(np.deg2rad(0.5)) * 3.0 / wt.sum() / 5.0
    np.testing.assert_allclose(float(O.l1_loss(p, t, w, std)), expect, rtol=1e-12)


def test_face_thickness_is_the_thinner_cell():
    lat = np.arange(-89.5, 90.0)
    h = np.full((O.NLAT, O.NLON, 2), 10.0)
    h[50, 7, 1] = 4.0                                        # partial bottom cell
    h[60, 0, 1] = 0.0                                        # below the seabed at the seam column
    b = O.base_weights(lat, h)
    cu = np.cos(np.deg2rad(lat[50]))
    np.testing.assert_allclose(b["u"][50, 7, 1], cu * 4.0)   # west face of the partial cell
    np.testing.assert_allclose(b["u"][50, 8, 1], cu * 4.0)   # east face
    np.testing.assert_allclose(b["u"][50, 9, 1], cu * 10.0)
    assert b["u"][60, 0, 1] == 0.0 and b["u"][60, 1, 1] == 0.0   # seam face and its neighbour
    assert b["v"][50, 7, 1] < b["v"][49, 7, 1] * 0.5 and b["v"][51, 7, 1] < b["v"][52, 7, 1] * 0.5
    assert b["u"][:, -1].sum() == 0.0                        # seam face counted once


def test_field_std_pooled_and_eta_mean_removed():
    w = {k: np.ones(3) for k in O.FIELDS}
    f1 = {k: np.array([0.0, 0.0, 0.0]) for k in O.FIELDS}
    f2 = {k: np.array([2.0, 2.0, 2.0]) for k in O.FIELDS}
    s = O.field_std([f1, f2], [w, w])
    assert np.isclose(s["T"], 1.0)                           # pooled over both days
    assert s["eta"] == 0.0                                   # each day's eta mean removed first



def test_gate_refuses_one_nan_sample():
    assert T.gate_failure(1.0, {"A_h": 0.1, "K_v": -2.0}) is None
    assert "K_v" in T.gate_failure(1.0, {"A_h": 0.1, "K_v": float("nan")})
    assert "nan" in T.gate_failure(float("nan"), {"A_h": 0.1})


def test_d20_mae_ignores_dry_nan_columns():
    m = np.array([[[[10.0, np.nan]]]]); o = np.array([[[[12.0, np.nan]]]])
    keep = np.array([[[[True, False]]]]); cosw = np.ones((1, 2))
    np.testing.assert_allclose(T.d20_mae(m, o, keep, cosw), [2.0])


def test_level_without_oras5_data_copies_level_above_and_is_masked():
    a = _day()
    a[O.NLEV - 1] = np.nan                                   # deepest T level: no data anywhere
    a[O.NLEV - 2] = 4.0
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = np.ones((O.NLAT, O.NLON + 1)), np.ones((O.NLAT + 1, O.NLON))
    f, v = O.day_to_fields(a, wet3, um, vm)
    assert np.isfinite(f["T"]).all() and np.allclose(f["T"][..., -1], f["T"][..., -2])
    assert not v["T"][..., -1].any()
