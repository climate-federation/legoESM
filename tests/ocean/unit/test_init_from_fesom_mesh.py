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


def _expected(lat_t, lon_t, z, lat_deg, lon_deg, T, nlev, k=4):
    """Independent per-layer wet-donor IDW + vertical linear interpolation."""
    def unit(la, lo):
        la, lo = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    d = np.linalg.norm(unit(lat_deg, lon_deg) - unit(lat_t, lon_t), axis=1)
    zs = 5.0 * np.arange(69) + 2.5

    def idw(kk):
        wet = np.flatnonzero(nlev - 2 >= kk)
        assert wet.size > 0
        j = wet[np.argsort(d[wet])[:min(k, wet.size)]]
        w = np.where(d[j] == 0.0, 1.0, 0.0) if (d[j] == 0.0).any() else d[j] ** -2.0
        return (w * T[j, kk]).sum() / w.sum()
    lo = int(np.clip(np.searchsorted(zs, z, side="right") - 1, 0, 68))
    hi = min(lo + 1, 68)
    f = 0.0 if hi == lo else np.clip((z - zs[lo]) / (zs[hi] - zs[lo]), 0.0, 1.0)
    return (1.0 - f) * idw(lo) + f * idw(hi)


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


def test_shallow_node_takes_deep_wet_donors_below_its_bottom(tmp_path):
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    i = int(np.argmax(nlev == 5))                        # 4 wet layers, bottom mid 17.5 m
    Ti, _ = init_ocean_from_fesom_mesh(_grid(lat_deg[i], lon_deg[i]), _z_coord(), d)
    Ti = np.asarray(Ti)
    np.testing.assert_allclose(Ti[0, 0], T[i, 0], rtol=0.0, atol=1e-9)      # surface: itself
    exp = _expected(lat_deg[i], lon_deg[i], 3000.0, lat_deg, lon_deg, T, nlev)
    np.testing.assert_allclose(Ti[0, 11], exp, rtol=0.0, atol=1e-9)         # 3000 m: deep donors
    assert abs(exp - T[i, 3]) > 1e-3                     # not the shallow bottom value extended
    assert exp != 0.0


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
    assert any("1 isolated wet target cells" in m for m in msgs)
    msgs = []
    init_ocean_from_fesom_mesh(_grid(lat, lon), _z_coord(), d,
                               isolated_factor=1e9, log=msgs.append)
    assert any("0 isolated wet target cells" in m for m in msgs)
    msgs = []
    init_ocean_from_fesom_mesh(_grid(lat, lon), _z_coord(), d,
                               wet_mask=np.array([True, False]), log=msgs.append)
    assert any("0 isolated wet target cells" in m for m in msgs)   # the far cell is land


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
    S = np.where(T != 0.0, 35.0 + 0.01 * lon_deg[:, None], 0.0)
    for col, z in ((0, 2.5), (3, 20.0), (11, 3000.0)):
        np.testing.assert_allclose(T_res[0, col], _expected(45.3, 7.6, z, lat_deg, lon_deg, T, nlev), atol=1e-9)
        np.testing.assert_allclose(S_res[0, col], _expected(45.3, 7.6, z, lat_deg, lon_deg, S, nlev), atol=1e-9)
    # discriminates against a nearest-only remap and against a column-extension remap
    def unit(la, lo):
        la, lo = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    dist = np.linalg.norm(unit(lat_deg, lon_deg) - unit(45.3, 7.6), axis=1)
    near = int(np.argmin(dist))
    zs = 5.0 * np.arange(69) + 2.5
    assert abs(T_res[0, 11] - np.interp(3000.0, zs[:nlev[near] - 1], T[near, :nlev[near] - 1])) > 1e-6


def test_layers_with_one_or_zero_wet_donors(tmp_path):
    """Single-donor layers query k=1 (1-D result), empty layers hold the layer above;
    a padded 0.0 must never enter the average."""
    d, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    nlev2 = np.minimum(nlev, 60)                   # nobody wet beyond layer 58 ...
    nlev2[0] = 66                                  # ... except node 0, wet through layer 64
    zabs = 5.0 * np.arange(70) + 2.5
    T2 = np.where(np.arange(70)[None, :] <= (nlev2[:, None] - 2),
                  10.0 - 0.002 * zabs + 0.1 * lat_deg[:, None], 0.0)
    np.save(d / "nlevels_nod2D.npy", nlev2.astype(np.int32))
    np.save(d / "T_ic.npy", T2)
    zc = SimpleNamespace(n_levels=3, z_full_ref=-np.array([5.0 * 60 + 2.5, 5.0 * 64 + 2.5, 3000.0]))
    Ti, _ = init_ocean_from_fesom_mesh(_grid(lat_deg[5], lon_deg[5]), zc, d)
    Ti = np.asarray(Ti)[0]
    np.testing.assert_allclose(Ti[0], T2[0, 60], atol=1e-9)   # layer 60: only node 0 is wet
    np.testing.assert_allclose(Ti[1], T2[0, 64], atol=1e-9)   # layer 64: only node 0 is wet
    np.testing.assert_allclose(Ti[2], T2[0, 64], atol=1e-9)   # layers 65..68 empty: hold 64
    assert Ti[2] != 0.0
    msgs = []
    init_ocean_from_fesom_mesh(_grid(lat_deg[5], lon_deg[5]), zc, d,
                               wet_mask=np.array([False]), log=msgs.append)  # all-land: no crash
    assert any("0 isolated" in m for m in msgs)


def test_cache_hit_is_bit_identical_miss_on_new_key_and_nonzero_rank_waits(tmp_path, monkeypatch):
    import jax
    import legoesm.ocean.init_woa as init_woa

    mesh_dir, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    grid = _grid([44.0, 47.5, 50.0], [3.0, 7.5, 12.0])
    cache_dir = tmp_path / "c"
    logs = []
    T1, S1 = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir,
                                        cache_dir=str(cache_dir), log=logs.append)
    written = sorted(p.name for p in cache_dir.glob("*.npz"))
    assert len(written) == 1 and not list(cache_dir.glob("*.tmp.*"))
    assert any("cache written" in m for m in logs)
    logs.clear()
    T2, S2 = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir,
                                        cache_dir=str(cache_dir), log=logs.append)
    assert any("cache hit" in m for m in logs)
    assert np.asarray(T1).tobytes() == np.asarray(T2).tobytes()
    assert np.asarray(S1).tobytes() == np.asarray(S2).tobytes()
    # no cache at all -> a fresh build gives the same bytes the cache served
    monkeypatch.delenv("LEGOESM_MESH_CACHE_DIR", raising=False)
    T3, _ = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, cache_dir=None, log=logs.append)
    assert np.asarray(T1).tobytes() == np.asarray(T3).tobytes()
    with pytest.raises(ValueError, match="absolute"):
        init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, cache_dir="rel/c", log=logs.append)

    # a different k hashes to a different filename -> a miss, not a hit
    logs.clear()
    init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, k=5,
                               cache_dir=str(cache_dir), log=logs.append)
    written2 = sorted(p.name for p in cache_dir.glob("*.npz"))
    assert len(written2) == 2 and written[0] in written2
    assert not any("cache hit" in m for m in logs)

    # a non-zero rank never builds: it waits for rank 0's file and times out
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(init_woa, "_FESOM_IC_CACHE_POLL_S", 0.01)
    with pytest.raises(RuntimeError, match="timed out"):
        init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, cache_dir=str(tmp_path / "c2"),
                                   cache_wait_timeout_s=0.1, log=logs.append)
    assert not list((tmp_path / "c2").glob("*.npz"))


def test_cache_corrupt_file_rebuilds_on_rank0_and_loads_on_rank1(tmp_path, monkeypatch):
    import glob
    import os
    import jax

    cache_dir = str(tmp_path / "cache")
    mesh_dir, lat_deg, lon_deg, T, nlev = _write_mesh(tmp_path)
    grid = _grid([44.0, 47.5, 50.0], [3.0, 7.5, 12.0])
    messages = []

    def run(rank):
        monkeypatch.setattr(jax, "process_index", lambda: rank)
        monkeypatch.setattr(jax, "process_count", lambda: 2)
        messages.clear()
        Tj, Sj = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, cache_dir=cache_dir,
                                            log=messages.append, cache_wait_timeout_s=30.0)
        return np.asarray(Tj), np.asarray(Sj)

    T0, S0 = run(rank=0)
    (cache_path,) = glob.glob(os.path.join(cache_dir, "fesom_ic_v*.npz"))
    assert not glob.glob(os.path.join(cache_dir, "*.tmp.*"))

    # (1) corrupt entry: rank 0 logs why, rebuilds, and leaves a valid file
    with open(cache_path, "wb") as fh:
        fh.write(b"definitely not an npz")
    T0b, S0b = run(rank=0)
    assert any("rebuild" in m for m in messages)
    assert T0b.tobytes() == T0.tobytes() and S0b.tobytes() == S0.tobytes()
    with np.load(cache_path) as blob:
        assert blob["T"].shape == T0.shape and blob["S"].shape == S0.shape

    # (2) a peer rank loads the valid cache: bit-identical, no rebuild
    T1, S1 = run(rank=1)
    assert any("cache hit" in m for m in messages)
    assert not any("rebuild" in m for m in messages)
    assert T1.tobytes() == T0.tobytes() and S1.tobytes() == S0.tobytes()

    # (3) peer rank + corrupt entry: fail fast instead of polling
    with open(cache_path, "wb") as fh:
        fh.write(b"still not an npz")
    with pytest.raises(RuntimeError, match="corrupt"):
        run(rank=1)


def test_cache_hit_does_not_rebuild(tmp_path):
    import os
    import shutil

    mesh_dir, lat_deg, lon_deg, _T_mesh, _nlev = _write_mesh(tmp_path)
    grid = _grid([44.0, 47.5, 50.0], [3.0, 7.5, 12.0])
    cache_dir = tmp_path / "cache"
    logs = []
    T_built, S_built = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir,
                                                  cache_dir=str(cache_dir), log=logs.append)
    assert any(cache_dir.iterdir())

    # Corrupt the mesh file in place while preserving everything the stat-based
    # key sees (size, mtime_ns): a second call can succeed only via a hit.
    t_ic = os.path.join(mesh_dir, "T_ic.npy")
    st = os.stat(t_ic)
    n = st.st_size
    with open(t_ic, "wb") as fh:
        fh.write(b"CORRUPTED-NOT-AN-NPY" + b"\x00" * (n - 20))
    os.utime(t_ic, ns=(st.st_atime_ns, st.st_mtime_ns))
    T_hit, S_hit = init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir,
                                              cache_dir=str(cache_dir), log=logs.append)
    assert np.asarray(T_hit).tobytes() == np.asarray(T_built).tobytes()
    assert np.asarray(S_hit).tobytes() == np.asarray(S_built).tobytes()

    # with the cache gone the corrupted mesh must fail on the build path,
    # proving the call above bypassed the build
    shutil.rmtree(cache_dir)
    with pytest.raises(Exception):
        init_ocean_from_fesom_mesh(grid, _z_coord(), mesh_dir, cache_dir=str(cache_dir),
                                   log=logs.append)
