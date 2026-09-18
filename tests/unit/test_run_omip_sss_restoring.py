"""FESOM2-style SSS restoring on run_omip: monthly PHC2 target by calendar
day, area-weighted mean removal, and the per-block reference selection."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import scripts.run.run_omip as run_omip  # noqa: E402


def test_month_index_noleap():
    # day 58 = 28 Feb, day 59 = 1 Mar on the noleap calendar
    got = [run_omip._sss_month_index(d) for d in (0, 30, 31, 58, 59, 364, 365, 400)]
    assert got == [0, 0, 1, 1, 2, 11, 0, 1]


def test_restore_sss_top_matches_old_formula_and_removes_mean():
    rng = np.random.default_rng(0)
    S = jnp.asarray(rng.uniform(33, 37, (4, 6, 3)))
    target = jnp.asarray(rng.uniform(33, 37, (4, 6)))
    mask = jnp.asarray(rng.integers(0, 2, (4, 6)).astype(float))
    area = jnp.asarray(rng.uniform(1, 3, (4, 6)))
    alpha = 0.01
    old = S.at[..., 0].set(S[..., 0] - alpha * (S[..., 0] - target) * mask)
    new = run_omip._restore_sss_top(S, target, alpha, mask)
    assert np.array_equal(np.asarray(old), np.asarray(new))
    bal = run_omip._restore_sss_top(S, target, alpha, mask, area=area, remove_mean=True)
    dS = np.asarray(bal[..., 0] - S[..., 0])
    assert np.all(dS[np.asarray(mask) == 0] == 0.0)
    assert abs(float((np.asarray(area) * dS).sum())) <= 1e-12 * float((np.asarray(area) * np.abs(dS)).sum())
    assert np.any(dS != 0.0)
    with pytest.raises(ValueError, match="area"):
        run_omip._restore_sss_top(S, target, alpha, mask, remove_mean=True)


def test_refs_for_block_selects_month_and_caches_shards():
    base = {"sponge_gamma": jnp.zeros(5)}
    assert run_omip._refs_for_block(base, {}, 40.0) is base
    monthly = jnp.arange(12.0)[:, None] * jnp.ones((1, 5))
    st = {"sss_target_monthly": monthly, "sss_remove_mean": True, "sss_area": jnp.ones(5)}
    refs = run_omip._refs_for_block(base, st, 40.0)
    assert "sponge_gamma" in refs and np.all(np.asarray(refs["sss_target"]) == 1.0)
    assert np.all(np.asarray(refs["sss_area"]) == 1.0)
    calls = []

    def shard(x):
        calls.append(1)
        return x * 2

    r1 = run_omip._refs_for_block(None, st, 40.0, shard_fn=shard)
    r2 = run_omip._refs_for_block(None, st, 100.0, shard_fn=shard)
    assert len(calls) == 13                       # 12 months + area, once
    assert np.all(np.asarray(r1["sss_target"]) == 2.0)
    assert np.all(np.asarray(r2["sss_target"]) == 6.0)   # day 100 -> April (index 3)


def _write_phc2_like(path, lat, lon):
    nc = pytest.importorskip("netCDF4")
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 12); ds.createDimension("lat", lat.size); ds.createDimension("lon", lon.size)
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        v = ds.createVariable("SALT", "f4", ("time", "lat", "lon"))
        v.missing_value = np.float32(-99.0)
        # spatially varying so a donor/permutation error is visible
        data = (35.0 + np.arange(12.0)[:, None, None]
                + 0.1 * np.arange(lat.size)[None, :, None]
                + 0.01 * np.arange(lon.size)[None, None, :])
        data[:, 1, 2] = -99.0
        v[:] = data


def test_load_phc2_monthly_fills_and_checks_axes(tmp_path):
    from legoesm.grids.regridding import RegridWeights

    lat = np.array([-1.5, -0.5, 0.5, 1.5]); lon = 0.5 + np.arange(8.0)
    f = tmp_path / "phc2.nc"
    _write_phc2_like(f, lat, lon)
    rw = RegridWeights(src_indices=jnp.arange(32)[:, None], weights=jnp.ones((32, 1)),
                       target_shape=(4, 8), src_flat_size=32)
    out = run_omip._load_phc2_monthly_sss_target(str(f), rw, lat, lon)
    assert out.shape == (12, 4, 8)
    assert np.all(np.isfinite(out))
    expect = (35.0 + np.arange(12.0)[:, None, None] + 0.1 * np.arange(4)[None, :, None]
              + 0.01 * np.arange(8)[None, None, :])
    valid = np.ones((4, 8), dtype=bool); valid[1, 2] = False
    assert np.allclose(out[:, valid], expect[:, valid])        # identity weights: no permutation
    neighbours = [expect[:, 1, 1], expect[:, 1, 3], expect[:, 0, 2], expect[:, 2, 2]]
    assert any(np.allclose(out[:, 1, 2], nb) for nb in neighbours)   # filled from a 4-neighbour
    with pytest.raises(ValueError, match="lat/lon"):
        run_omip._load_phc2_monthly_sss_target(str(f), rw, lat + 0.25, lon)


def test_cli_round_trip():
    a = run_omip.parse_args(["--sss-restoring-target", "phc2_monthly",
                             "--sss-target-file", "/x/y.nc", "--sss-restoring-remove-mean",
                             "--sss-piston-velocity", "1.929e-6"])
    assert a.sss_restoring_target == "phc2_monthly"
    assert a.sss_target_file == "/x/y.nc"
    assert a.sss_restoring_remove_mean is True
    assert a.sss_piston_velocity == 1.929e-6
    d = run_omip.parse_args([])
    assert d.sss_restoring_target == "woa_winter" and d.sss_restoring_remove_mean is False
    with pytest.raises(SystemExit):
        run_omip.parse_args(["--sss-restoring-target", "bogus"])


def test_builders_treat_every_ref_key_as_optional():
    """A per-block refs dict may carry only the SSS entries (monthly target on
    a non-sharded lane); both block builders must not assume the sponge keys."""
    import inspect

    src_a = inspect.getsource(run_omip._build_jra55_block_fn)
    src_b = inspect.getsource(run_omip._build_jra55_block_fn_interp)
    for src in (src_a, src_b):
        assert 'if enable_sponge and "sponge_gamma" in refs:' in src
        assert 'if enable_sss and "sss_area" in refs:' in src


def test_phc2_target_goes_through_the_host_regrid(tmp_path, monkeypatch):
    """Route-B multicontroller: the GPU regrid differed byte-wise across
    processes, so the target must take the deterministic host regrid."""
    import inspect

    from legoesm.grids.regridding import RegridWeights

    lat = np.array([-1.5, -0.5, 0.5, 1.5]); lon = 0.5 + np.arange(8.0)
    f = tmp_path / "phc2.nc"
    _write_phc2_like(f, lat, lon)
    rw = RegridWeights(src_indices=jnp.arange(32)[:, None], weights=jnp.ones((32, 1)),
                       target_shape=(4, 8), src_flat_size=32)
    recorded = []

    def stub(recs, rw_):
        recorded.append(np.asarray(recs).copy())
        return np.zeros((12,) + tuple(rw_.target_shape))

    monkeypatch.setattr(run_omip, "_regrid_records_host", stub)
    out = run_omip._load_phc2_monthly_sss_target(str(f), rw, lat, lon)
    assert len(recorded) == 1 and recorded[0].shape == (12, 4, 8)
    assert not np.isnan(recorded[0]).any()
    assert out.shape == (12, 4, 8)
    # the PHC2 wrapper delegates to the shared monthly-climatology loader; pin
    # the symbol that executes the regrid
    src = inspect.getsource(run_omip._load_phc2_monthly_sss_target)
    assert "_load_monthly_clim_target(" in src
    src = inspect.getsource(run_omip._load_monthly_clim_target)
    assert "_regrid_records_host(" in src and "regrid_scalar(" not in src


def test_both_spmd_branches_set_the_ref_sharder():
    import inspect

    src = inspect.getsource(run_omip.run_omip_single)
    i = src.index("spmd_shard_stack = partial(shard_cell_stack_spmd")
    assert "spmd_shard_ref = spmd_shard_stack" in src[i:i + 500]
    assert "spmd_shard_ref = partial(shard_forcing_latlon" in src
