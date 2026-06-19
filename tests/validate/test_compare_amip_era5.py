"""Unit tests for the ``scripts/validate/compare_amip_era5.py`` driver helpers.

The ``main`` CLI does live I/O (restart + ERA5 GCS) and is not unit-tested
here, but every importable helper is: grid-type dispatch (raise on unknown),
ERA5-regrid selection, sigma/lat-lon extraction, restart→ColumnState assembly,
and the end-to-end compare_and_write on synthetic states (writing a temp
manifest).  This pins the Stage-2 wiring contract per CLAUDE.md (every new
``.py`` incl. scripts gets a direct test; dispatch raises on unknown).
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from scripts.validate import compare_amip_era5 as drv


def test_canonical_grid_type_aliases():
    assert drv.canonical_grid_type("cubed_sphere") == "cubed_sphere"
    assert drv.canonical_grid_type("CS") == "cubed_sphere"
    assert drv.canonical_grid_type("gaussian") == "spectral"
    assert drv.canonical_grid_type("lat_lon") == "latlon"


def test_canonical_grid_type_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown grid_type"):
        drv.canonical_grid_type("octahedral")


def test_select_era5_regrid_dispatch():
    # Returns the matching era5_to_*_carry callable; unknown raises.
    f_cs = drv.select_era5_regrid("cubed_sphere")
    f_ll = drv.select_era5_regrid("latlon")
    f_sp = drv.select_era5_regrid("gaussian")
    assert f_cs.__name__ == "era5_to_cubedsphere_carry"
    assert f_ll.__name__ == "era5_to_latlon_carry"
    assert f_sp.__name__ == "era5_to_spectral_carry"
    with pytest.raises(ValueError):
        drv.select_era5_regrid("nope")


class _FakeSigma:
    def __init__(self, nlev):
        half = jnp.linspace(0.0, 1.0, nlev + 1)
        self.sigma_half = half
        self.sigma_full = 0.5 * (half[1:] + half[:-1])


def test_sigma_levels_extraction():
    sig = _FakeSigma(5)
    sf, sh = drv.sigma_levels(sig)
    assert sf.shape == (5,)
    assert sh.shape == (6,)
    assert float(sh[0]) == 0.0 and float(sh[-1]) == 1.0


class _FakeGrid:
    """Uniform grid_lat/grid_lon accessors (radians), grid-shaped."""

    def __init__(self, lat_rad, lon_rad):
        self.grid_lat = lat_rad
        self.grid_lon = lon_rad


def test_grid_lat_lon_deg_uniform_accessor():
    import numpy as np

    lat = jnp.deg2rad(jnp.array([[0.0, 10.0], [20.0, 30.0]]))
    lon = jnp.deg2rad(jnp.array([[100.0, 110.0], [120.0, 130.0]]))
    grid = _FakeGrid(lat, lon)
    lat_d, lon_d = drv.grid_lat_lon_deg(grid)
    # The accessor matches the CANONICAL rad→deg of the same inputs at the grid's
    # NATIVE precision (float32 by default). The previous literal abs=1e-6 was tighter
    # than the float32 deg→rad→deg roundtrip (≈1.9e-6 deg at 30°), so it passed ONLY
    # under the x64 leak from another test file (jax_enable_x64 set at import) — an
    # order-dependent flaky test that fails in isolation / under xdist sharding (iter 108).
    np.testing.assert_allclose(np.asarray(lat_d), np.rad2deg(np.asarray(lat)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(lon_d), np.rad2deg(np.asarray(lon)), rtol=1e-6)
    # the [1,1] / [0,1] cells are 30° / 110° (float32-safe tolerance).
    assert float(lat_d[1, 1]) == pytest.approx(30.0, abs=1e-4)
    assert float(lon_d[0, 1]) == pytest.approx(110.0, abs=1e-4)


class _FakeState:
    def __init__(self, shape, nlev):
        full = shape + (nlev,)
        self.T = jnp.full(full, 250.0)
        self.u = jnp.full(full, 5.0)
        self.v = jnp.zeros(full)
        self.p_s = jnp.full(shape, 1.0e5)


def test_model_state_from_restart():
    state = _FakeState((2, 2), 4)
    q_v = jnp.full((2, 2, 4), 1e-3)
    cs = drv.model_state_from_restart(
        state, q_v, sst_K=jnp.full((2, 2), 300.0)
    )
    assert cs.T.shape == (2, 2, 4)
    assert cs.q_v.shape == (2, 2, 4)
    assert float(cs.sst_K[0, 0]) == pytest.approx(300.0)
    assert cs.precip_mm_day is None


def _stub_main_io(monkeypatch, *, nlev=4, shape=(2, 2)):
    """Stub all of main()'s live I/O (grid factory, sigma, restart, ERA5, regrid)
    and return ``(grid, calls)``.  ``calls['n_load_restart']`` counts restart loads
    (the before/after path loads TWO).  Shared by the wiring + before/after tests."""
    from types import SimpleNamespace

    sig = _FakeSigma(nlev)
    lat = jnp.deg2rad(jnp.array([[0.0, 0.0], [1.0, 1.0]]))
    lon = jnp.deg2rad(jnp.array([[0.0, 1.0], [0.0, 1.0]]))

    class _Grid(_FakeGrid):
        grid_shape_2d = shape
        grid_area = jnp.ones(shape)   # real grids expose grid_area (GridProtocol)

    grid = _Grid(lat, lon)
    state = _FakeState(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 1e-3)
    calls = {"n_load_restart": 0}

    def fake_create_grid(token, resolution=None):
        calls["create_grid"] = (token, resolution)
        return grid

    def fake_create_sigma(nlev_arg):
        calls["nlev"] = nlev_arg
        return sig

    def fake_load_restart(path, g, s, strict=True):
        calls["load_restart"] = (path, strict)
        calls["n_load_restart"] += 1
        # 10-tuple: (state, q_v, step, day, config, diag, q_c, q_r, meta, aux)
        return (state, q_v, 0, 0.0, None, None, None, None, None, None)

    class _FakeERA5Cfg:
        def __init__(self, **kw):
            calls["era5_cfg"] = kw

    def fake_load_slice(cfg, time_idx):
        calls["era5_time_idx"] = time_idx
        return "ERA5SLICE"

    def fake_regrid(slice_obj, g, s):
        calls["regrid"] = (slice_obj, g is grid, s is sig)
        return SimpleNamespace(T=state.T, q_v=q_v, u=state.u, v=state.v, p_s=state.p_s)

    import legoesm.driver.restart as restart
    import legoesm.grids.factory as factory
    import legoesm.grids.vertical as vertical
    import legoesm.training.era5_to_state as e2s

    monkeypatch.setattr(factory, "create_grid", fake_create_grid)
    monkeypatch.setattr(vertical, "create_sigma_coordinate", fake_create_sigma)
    monkeypatch.setattr(restart, "load_restart", fake_load_restart)
    monkeypatch.setattr(e2s, "TrainingERA5Config", _FakeERA5Cfg)
    monkeypatch.setattr(e2s, "load_era5_slice", fake_load_slice)
    monkeypatch.setattr(drv, "select_era5_regrid", lambda gt: fake_regrid)
    return grid, calls


def test_main_wiring_monkeypatched(tmp_path, monkeypatch):
    """Drive main() with all live I/O stubbed; assert the wiring contract.

    Verifies the factory token mapping (spectral->gaussian), create_sigma_coord
    nlev, load_restart strict=True + correct unpack, regrid call order
    (slice, grid, sigma), and that the manifest is written.
    """
    nlev = 4
    _grid, calls = _stub_main_io(monkeypatch, nlev=nlev)

    out = str(tmp_path / "m.json")
    with pytest.warns(UserWarning, match="surface air temperature"):
        rc = drv.main([
            "--restart", "chk.npz", "--grid-type", "gaussian",
            "--resolution", "8", "--nlev", str(nlev),
            "--era5-zarr", "gs://x", "--era5-time-idx", "3",
            "--n-worst", "1", "--out", out,
        ])
    assert rc == 0
    assert calls["create_grid"] == ("gaussian", 8)  # spectral -> gaussian token
    assert calls["nlev"] == nlev
    assert calls["load_restart"][1] is True  # strict=True
    assert calls["n_load_restart"] == 1      # no baseline ⇒ single load
    assert calls["era5_time_idx"] == 3
    assert calls["regrid"] == ("ERA5SLICE", True, True)  # (slice, grid, sigma)
    with open(out) as f:
        dicts = json.load(f)
    assert len(dicts) == 1


def test_main_before_after_baseline_restart(tmp_path, monkeypatch, capsys):
    """--baseline-restart drives the per-variable before/after path: a SECOND restart
    is loaded and the per-variable global-bias change is printed (clause-5 check).
    The stub returns the SAME state for both, so baseline==corrected and the print
    reports no improvement — the wiring (two loads + per_variable_bias_improvement +
    the print loop) runs end-to-end."""
    nlev = 4
    _grid, calls = _stub_main_io(monkeypatch, nlev=nlev)

    out = str(tmp_path / "m.json")
    with pytest.warns(UserWarning, match="surface air temperature"):
        rc = drv.main([
            "--restart", "corrected.npz", "--baseline-restart", "baseline.npz",
            "--grid-type", "gaussian", "--resolution", "8", "--nlev", str(nlev),
            "--era5-zarr", "gs://x", "--n-worst", "1", "--out", out,
        ])
    assert rc == 0
    assert calls["n_load_restart"] == 2      # corrected + baseline both loaded
    printed = capsys.readouterr().out
    assert "per-variable bias baseline -> corrected" in printed
    assert "T_rmse" in printed and "wind_rmse" in printed
    # baseline == corrected (same stub state) ⇒ equal bias ⇒ NOT improved.
    assert "WORSE/same" in printed


def test_compare_and_write_end_to_end(tmp_path):
    nlev = 5
    shape = (2, 2)
    sig = _FakeSigma(nlev)
    lat = jnp.deg2rad(jnp.array([[0.0, 0.0], [1.0, 1.0]]))
    lon = jnp.deg2rad(jnp.array([[0.0, 1.0], [0.0, 1.0]]))
    grid = _FakeGrid(lat, lon)

    state = _FakeState(shape, nlev)
    q_v = jnp.full(shape + (nlev,), 1e-3)
    model = drv.model_state_from_restart(
        state, q_v, sst_K=jnp.full(shape, 300.0)
    )
    # Reference identical except a cold bias in one column.
    import numpy as np

    T_ref = np.full(shape + (nlev,), 250.0)
    T_ref[1, 0] = 245.0  # 5 K bias at (1,0)
    reference = model._replace(T=jnp.asarray(T_ref), sst_K=None)

    out = str(tmp_path / "worst.json")
    result = drv.compare_and_write(
        model=model, reference=reference, sigma=sig, grid=grid,
        time_index=2, n_worst=1, out_path=out,
    )
    assert len(result.manifest) == 1
    assert result.manifest[0].grid_index == (1, 0)
    # Manifest written and reloadable.
    with open(out) as f:
        dicts = json.load(f)
    assert dicts[0]["grid_index"] == [1, 0]
    assert dicts[0]["time_index"] == 2
