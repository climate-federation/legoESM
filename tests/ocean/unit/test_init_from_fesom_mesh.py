"""Unit tests for legoesm.ocean.init_woa.init_ocean_from_fesom_mesh."""
import numpy as np
import pytest
from types import SimpleNamespace

from legoesm.ocean.init_woa import init_ocean_from_fesom_mesh

_Z_CENTER = np.array([2.5, 7.5, 12.5, 20.0, 60.0, 100.0, 300.0,
                      600.0, 1000.0, 1500.0, 2000.0, 3000.0])


def _write_mesh(tmp_path):
    """Write a synthetic FESOM mesh; returns (mesh_dir, lat_deg, lon_deg, T, nlevels)."""
    lon_deg, lat_deg = np.meshgrid(np.arange(15.0), np.arange(40.0, 55.0))
    lon_deg, lat_deg = lon_deg.ravel(), lat_deg.ravel()
    n2 = lon_deg.size
    nlev = 5 + (np.arange(n2) % 66)                      # 5..70, so 4..69 wet layers
    zabs = 5.0 * np.arange(70) + 2.5                     # |Z_k| = 5k + 2.5
    wet = np.arange(70)[None, :] <= (nlev[:, None] - 2)  # layers 0..nlevels-2 are wet
    T = np.where(wet, 10.0 - 0.002 * zabs + 0.1 * lat_deg[:, None], 0.0)
    S = np.where(wet, 35.0 + 0.01 * lon_deg[:, None], 0.0)
    d = tmp_path / "mesh"
    d.mkdir()
    np.save(d / "T_ic.npy", T)
    np.save(d / "S_ic.npy", S)
    np.save(d / "geo_coord_nod2D.npy",
            np.stack([np.deg2rad(lon_deg), np.deg2rad(lat_deg)], axis=1))
    np.save(d / "Z.npy", -zabs[:69])
    np.save(d / "nlevels_nod2D.npy", nlev.astype(np.int32))
    np.save(d / "mesh_resolution.npy", np.full(n2, 1.1e5))
    return d, lat_deg, lon_deg, T, nlev


def _grid(lat_deg, lon_deg):
    return SimpleNamespace(
        latCell=np.deg2rad(np.atleast_1d(np.asarray(lat_deg, float))),
        lonCell=np.deg2rad(np.atleast_1d(np.asarray(lon_deg, float))))


def _z_coord():
    return SimpleNamespace(n_levels=12, z_full_ref=-_Z_CENTER.copy())


def test_coincident_deep_node_matches_layer_mids(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    i = int(np.argmax(nlev == 70))
    Ti, _ = init_ocean_from_fesom_mesh(_grid(lat_deg[i], lon_deg[i]), _z_coord(), d)
    Ti = np.asarray(Ti)
    assert Ti.shape == (1, 12)
    np.testing.assert_allclose(Ti[0, 0], T[i, 0], rtol=0.0, atol=1e-9)  # depth 2.5 = |Z_0|
    np.testing.assert_allclose(Ti[0, 2], T[i, 2], rtol=0.0, atol=1e-9)  # depth 12.5 = |Z_2|
    zc = SimpleNamespace(n_levels=2, z_full_ref=-np.array([0.5, 1.0]))   # above |Z_0|: top held
    Ts, _ = init_ocean_from_fesom_mesh(_grid(lat_deg[i], lon_deg[i]), zc, d)
    np.testing.assert_allclose(np.asarray(Ts)[0], T[i, 0], rtol=0.0, atol=1e-9)


def test_shallow_node_profile_extended_downward(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    i = int(np.argmax(nlev == 5))                        # deepest wet mid-depth 17.5 m
    Ti, _ = init_ocean_from_fesom_mesh(_grid(lat_deg[i], lon_deg[i]), _z_coord(), d)
    v = float(np.asarray(Ti)[0, 11])                     # target depth 3000 m
    np.testing.assert_allclose(v, T[i, 3], rtol=0.0, atol=1e-9)
    assert v != 0.0


def test_deep_node_interpolated_between_layers(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    i = int(np.argmax(nlev == 70))
    Ti, _ = init_ocean_from_fesom_mesh(_grid(lat_deg[i], lon_deg[i]), _z_coord(), d)
    expected = 0.5 * (T[i, 3] + T[i, 4])                 # 20 m splits 17.5 / 22.5 mids
    np.testing.assert_allclose(np.asarray(Ti)[0, 3], expected, rtol=0.0, atol=1e-9)


def test_shapes_dtype_and_finiteness(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    lat = np.concatenate([lat_deg, [45.0]])
    lon = np.concatenate([lon_deg, [100.0]])
    To, So = init_ocean_from_fesom_mesh(_grid(lat, lon), _z_coord(), d)
    To, So = np.asarray(To), np.asarray(So)
    assert To.shape == So.shape == (lat.size, 12)
    assert To.dtype == So.dtype == np.float64
    assert np.isfinite(To).all() and np.isfinite(So).all()


def test_isolated_target_count_in_log(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    i = int(np.argmax(nlev == 70))
    lat, lon = [lat_deg[i], 45.0], [lon_deg[i], 100.0]
    msgs = []
    init_ocean_from_fesom_mesh(_grid(lat, lon), _z_coord(), d, log=msgs.append)
    assert any("1 isolated target cells" in m for m in msgs)
    msgs = []
    init_ocean_from_fesom_mesh(_grid(lat, lon), _z_coord(), d,
                               isolated_factor=1e9, log=msgs.append)
    assert any("0 isolated target cells" in m for m in msgs)
    msgs = []
    init_ocean_from_fesom_mesh(_grid(lat, lon), _z_coord(), d,
                               wet_mask=np.array([True, False]), log=msgs.append)
    assert any("0 isolated target cells" in m for m in msgs)   # the far cell is land


def test_wrong_nlevels_length_raises(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    np.save(d / "nlevels_nod2D.npy", nlev[:-1].astype(np.int32))
    i = int(np.argmax(nlev == 70))
    with pytest.raises(ValueError):
        init_ocean_from_fesom_mesh(_grid(lat_deg[i], lon_deg[i]), _z_coord(), d)


def test_idw_over_four_nearest_donors_with_different_bottoms(tmp_path):
    mesh_dir, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    grid = _grid(np.array([45.3]), np.array([7.6]))  # single target, off any node
    T_res, S_res = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, k=4)
    T_res, S_res = np.asarray(T_res), np.asarray(S_res)
    assert T_res.shape == S_res.shape == (1, 12)

    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    nodes = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    tla, tlo = np.deg2rad(45.3), np.deg2rad(7.6)
    tgt = np.array([np.cos(tla) * np.cos(tlo), np.cos(tla) * np.sin(tlo), np.sin(tla)])
    d = np.linalg.norm(nodes - tgt, axis=1)   # chord distances on the unit sphere
    idx = np.argsort(d)[:4]                   # 4 nearest donors (distinct nlev/bottoms)
    assert len(set(nlev[idx])) > 1
    w = d[idx] ** -2.0
    w /= w.sum()
    zs = 5.0 * np.arange(69) + 2.5            # donor mid-depths |Z_k|

    def expected(z):
        return sum(w[j] * np.interp(z, zs[:nlev[n] - 1], T[n, :nlev[n] - 1])
                   for j, n in enumerate(idx))

    exp = np.array([expected(2.5), expected(20.0), expected(3000.0)])
    cols = [0, 3, 11]
    np.testing.assert_allclose(T_res[0, cols], exp, atol=1e-9)
    exp_s = sum(w[j] * (35.0 + 0.01 * lon_deg[n]) for j, n in enumerate(idx))
    np.testing.assert_allclose(S_res[0, cols], exp_s, atol=1e-9)
    near = idx[0]
    near_3000 = np.interp(3000.0, zs[:nlev[near] - 1], T[near, :nlev[near] - 1])
    assert abs(exp[2] - near_3000) > 1e-6  # weighted value differs from nearest donor alone
