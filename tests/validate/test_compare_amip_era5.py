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

    # pure-sigma layer pressures (the VerticalCoordProtocol the compare uses, iter 340).
    def pressure_at_full(self, p_s):
        return p_s[..., None] * self.sigma_full

    def pressure_at_half(self, p_s):
        return p_s[..., None] * self.sigma_half


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
        calls["coord"] = "sigma"
        return sig

    def fake_make_hybrid(nlev_arg, *, p_top_Pa=100.0, **kw):
        calls["nlev"] = nlev_arg          # default --vertical-coord is 'hybrid'
        calls["coord"] = "hybrid"
        calls["p_top_Pa"] = p_top_Pa
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
    monkeypatch.setattr(vertical, "make_hybrid_levels", fake_make_hybrid)
    monkeypatch.setattr(restart, "load_restart", fake_load_restart)
    monkeypatch.setattr(e2s, "TrainingERA5Config", _FakeERA5Cfg)
    monkeypatch.setattr(e2s, "load_era5_slice", fake_load_slice)
    monkeypatch.setattr(drv, "select_era5_regrid", lambda gt: fake_regrid)
    return grid, calls


def test_load_model_from_restart_shape_and_nlev_guards(monkeypatch):
    """load_model_from_restart fails LOUD when the restart's column shape OR nlev don't
    match the requested grid / --nlev (a wrong --grid-type/--resolution/--nlev) — not a
    silent garbage compare against ERA5. (grid_winds_from_spectral passes a non-spectral
    state through, so only load_restart is mocked.)"""
    from types import SimpleNamespace

    import legoesm.driver.restart as restart_mod

    grid = SimpleNamespace(grid_shape_2d=(2, 3))      # expect (2, 3) = 6 columns

    def _fake_restart(shape):
        def f(path, g, s, strict):                    # noqa: ARG001
            return (SimpleNamespace(T=jnp.zeros(shape)), jnp.zeros(shape))
        return f

    # Wrong column shape: (3, 3) != (2, 3).
    monkeypatch.setattr(restart_mod, "load_restart", _fake_restart((3, 3, 5)))
    with pytest.raises(ValueError, match="column shape"):
        drv.load_model_from_restart("x.npz", grid, object(), 5)
    # Right columns, wrong nlev: 7 != 5.
    monkeypatch.setattr(restart_mod, "load_restart", _fake_restart((2, 3, 7)))
    with pytest.raises(ValueError, match="restart nlev"):
        drv.load_model_from_restart("x.npz", grid, object(), 5)


def test_load_model_from_restart_vertical_coord_mismatch_raises(monkeypatch):
    """Pre-flight (iter 357): when the restart RECORDS the run's coordinate
    (loaded_config.grid.vertical_coord), load_model_from_restart fails LOUD if the
    operator's --vertical-coord disagrees — otherwise the state lands on the WRONG
    pressure levels and the before/after bias is silently wrong (the flag's help only
    WARNED).  Defensive: skipped when the recorded coordinate is unavailable."""
    from types import SimpleNamespace

    import legoesm.driver.restart as restart_mod

    grid = SimpleNamespace(grid_shape_2d=(2, 3))

    def _fake_restart_with_cfg(shape, run_vcoord, run_p_top=200.0):
        cfg = SimpleNamespace(grid=SimpleNamespace(
            vertical_coord=run_vcoord, p_top_Pa=run_p_top))
        def f(path, g, s, strict):                    # noqa: ARG001
            # 5-tuple: (state, q_v, step, day, loaded_config) — load_model reads [0],[1],[4].
            return (SimpleNamespace(T=jnp.zeros(shape)), jnp.zeros(shape), 0, 0.0, cfg)
        return f

    # MISMATCH: run was hybrid, operator passed sigma ⇒ raise BEFORE any shape work.
    monkeypatch.setattr(restart_mod, "load_restart", _fake_restart_with_cfg((2, 3, 5), "hybrid"))
    with pytest.raises(ValueError, match="does not match the restart's recorded run coordinate"):
        drv.load_model_from_restart("x.npz", grid, object(), 5, expected_vertical_coord="sigma")

    # MATCH: the coordinate agrees ⇒ the coord check PASSES and the SHAPE guard runs next
    # (wrong columns here ⇒ the shape error, proving the coord check did not fire).
    monkeypatch.setattr(restart_mod, "load_restart", _fake_restart_with_cfg((4, 4, 5), "hybrid"))
    with pytest.raises(ValueError, match="column shape"):
        drv.load_model_from_restart("x.npz", grid, object(), 5, expected_vertical_coord="hybrid")

    # HYBRID p_top MISMATCH: coord matches (hybrid) but the model top differs ⇒ raise.
    monkeypatch.setattr(restart_mod, "load_restart",
                        _fake_restart_with_cfg((2, 3, 5), "hybrid", run_p_top=50.0))
    with pytest.raises(ValueError, match="does not match the restart's recorded hybrid model top"):
        drv.load_model_from_restart("x.npz", grid, object(), 5,
                                    expected_vertical_coord="hybrid", expected_p_top_Pa=200.0)
    # Matching p_top ⇒ p_top check PASSES, proceeds to the shape guard.
    monkeypatch.setattr(restart_mod, "load_restart",
                        _fake_restart_with_cfg((4, 4, 5), "hybrid", run_p_top=200.0))
    with pytest.raises(ValueError, match="column shape"):
        drv.load_model_from_restart("x.npz", grid, object(), 5,
                                    expected_vertical_coord="hybrid", expected_p_top_Pa=200.0)


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
    assert calls["coord"] == "hybrid"        # --vertical-coord defaults to hybrid (iter 340)
    assert calls["load_restart"][1] is True  # strict=True
    assert calls["n_load_restart"] == 1      # no baseline ⇒ single load
    assert calls["era5_time_idx"] == 3
    assert calls["regrid"] == ("ERA5SLICE", True, True)  # (slice, grid, sigma)


def test_main_vertical_coord_sigma_builds_pure_sigma(tmp_path, monkeypatch):
    """--vertical-coord sigma builds the pure-sigma coordinate (vs the default hybrid), so a
    user verifying a pure-sigma run gets the matching coordinate for the model state, the
    ERA5 reference regrid, and the bias mass weights (iter 340)."""
    nlev = 4
    _grid, calls = _stub_main_io(monkeypatch, nlev=nlev)
    out = str(tmp_path / "m.json")
    with pytest.warns(UserWarning, match="surface air temperature"):
        rc = drv.main([
            "--restart", "chk.npz", "--grid-type", "gaussian",
            "--resolution", "8", "--nlev", str(nlev), "--vertical-coord", "sigma",
            "--era5-zarr", "gs://x", "--n-worst", "1", "--out", out,
        ])
    assert rc == 0
    assert calls["coord"] == "sigma"
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
    # baseline == corrected (same stub state) ⇒ equal COMBINED bias ⇒ NOT improved ⇒ the
    # held-out verify EXITS NON-ZERO (iter 288): an automated deploy&verify workflow must
    # detect a correction that did not generalize, not silently report success.
    assert rc == 1
    assert calls["n_load_restart"] == 2      # corrected + baseline both loaded
    printed = capsys.readouterr().out
    assert "per-variable bias baseline -> corrected" in printed
    assert "T_rmse" in printed and "wind_rmse" in printed
    assert "COMBINED bias" in printed        # the combined verdict the exit code gates on
    # baseline == corrected (same stub state) ⇒ equal bias ⇒ NOT improved.
    assert "WORSE/same" in printed


def test_held_out_verify_exits_zero_when_the_correction_improves(
        tmp_path, monkeypatch, capsys):
    """Complement of the no-improvement case (iter 288/289): when the CORRECTED model is
    closer to the held-out ERA5 than the BASELINE (combined bias FELL), the held-out
    verify EXITS 0 — so a deploy&verify workflow proceeds ONLY on a correction that
    GENERALIZED. The iter-288 test pins rc=1; this pins rc=0 so a 'return 1 always'
    regression is caught (both exit codes locked)."""
    import legoesm.driver.restart as restart

    nlev = 4
    shape = (2, 2)
    _grid, _calls = _stub_main_io(monkeypatch, nlev=nlev)
    good = _FakeState(shape, nlev)               # T=250 == the stub reference (fake_regrid)
    bad = _FakeState(shape, nlev)
    bad.T = good.T - 8.0                          # an 8 K cold bias vs the reference
    q_v = jnp.full(shape + (nlev,), 1e-3)

    def path_dependent_load(path, g, s, strict=True):  # noqa: ARG001
        st = bad if "baseline" in path else good     # corrected matches ref; baseline biased
        return (st, q_v, 0, 0.0, None, None, None, None, None, None)

    monkeypatch.setattr(restart, "load_restart", path_dependent_load)
    out = str(tmp_path / "m.json")
    with pytest.warns(UserWarning, match="surface air temperature"):
        rc = drv.main([
            "--restart", "corrected.npz", "--baseline-restart", "baseline.npz",
            "--grid-type", "gaussian", "--resolution", "8", "--nlev", str(nlev),
            "--era5-zarr", "gs://x", "--n-worst", "1", "--out", out])
    assert rc == 0                               # corrected (0 bias) < baseline (8 K) ⇒ improved
    printed = capsys.readouterr().out
    assert "COMBINED bias" in printed and "improved" in printed


def test_held_out_verify_fails_on_a_blown_up_correction(tmp_path, monkeypatch):
    """A correction that DESTABILISES the model (a NON-FINITE held-out state) must FAIL
    the held-out verification — exit NON-zero, gracefully (no crash, no false PASS) — so
    a deploy&verify workflow never proceeds on a blown-up correction.  The NaN corrected
    bias is not ``< baseline`` ⇒ not 'improved' ⇒ the gate rejects it.  (Locks the
    blow-up path alongside the iter-288/289 improve→0 / no-improve→1 cases.)"""
    import legoesm.driver.restart as restart

    nlev = 4
    shape = (2, 2)
    _grid, _calls = _stub_main_io(monkeypatch, nlev=nlev)
    blown = _FakeState(shape, nlev)
    blown.T = jnp.full_like(blown.T, jnp.nan)        # the corrected run blew up (all NaN)
    baseline = _FakeState(shape, nlev)
    baseline.T = baseline.T - 8.0                    # a FINITE biased baseline
    q_v = jnp.full(shape + (nlev,), 1e-3)

    def path_dependent_load(path, g, s, strict=True):  # noqa: ARG001
        st = baseline if "baseline" in path else blown
        return (st, q_v, 0, 0.0, None, None, None, None, None, None)

    monkeypatch.setattr(restart, "load_restart", path_dependent_load)
    out = str(tmp_path / "m.json")
    with pytest.warns(UserWarning, match="surface air temperature"):
        rc = drv.main([
            "--restart", "corrected.npz", "--baseline-restart", "baseline.npz",
            "--grid-type", "gaussian", "--resolution", "8", "--nlev", str(nlev),
            "--era5-zarr", "gs://x", "--n-worst", "1", "--out", out])
    # A non-finite corrected bias is NOT an improvement ⇒ the gate rejects (no false pass).
    assert rc != 0


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

    T_ref = np.full(shape + (nlev,), 250.0)  # noqa: N806 (T = temperature, physics symbol)
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


def test_ocean_mask_for_verify(tmp_path):
    """--ocean-only sources an ocean valid_mask from --base-config (iter 452): off => None;
    on without --base-config or on an all-land grid or a grid-shape mismatch fails loud;
    on with a 2-ocean/2-land land fraction returns the 2-ocean mask."""
    from types import SimpleNamespace

    import legoesm.driver.model_driver as md

    # off => None (no probe, grid irrelevant)
    off = SimpleNamespace(ocean_only=False, base_config="", max_land_fraction=0.5)
    assert drv._ocean_mask_for_verify(off, grid=None) is None

    # on but no --base-config => fail loud
    with pytest.raises(SystemExit, match="requires --base-config"):
        drv._ocean_mask_for_verify(
            SimpleNamespace(ocean_only=True, base_config="", max_land_fraction=0.5),
            grid=None)

    cfg_path = str(tmp_path / "base.json")
    with open(cfg_path, "w") as f:
        json.dump({}, f)                       # default ExperimentConfig (loader fills it)

    class _FakeDriver:
        def __init__(self, _cfg):
            pass

        def static_land_fraction(self):
            return jnp.array([[0.0, 0.6], [0.4, 1.0]])   # 2 ocean (<=0.5), 2 land

    grid4 = SimpleNamespace(grid_area=jnp.ones((2, 2)))   # 4 columns on a STRUCTURED 2x2 grid
    grid9 = SimpleNamespace(grid_area=jnp.ones(9))        # mismatch
    on = SimpleNamespace(ocean_only=True, base_config=cfg_path, max_land_fraction=0.5)

    orig = md.ModelDriver
    md.ModelDriver = _FakeDriver
    try:
        mask = drv._ocean_mask_for_verify(on, grid4)
        assert mask is not None and int(jnp.sum(mask)) == 2
        # REGRESSION (iter 484): the mask must be GRID-SHAPED (broadcastable to the 2D
        # per-column score in aggregate_combined_bias), NOT a flat (n_columns,) array — a flat
        # mask would crash the bias reduction on a structured grid.
        assert mask.shape == (2, 2)

        with pytest.raises(SystemExit, match="must match"):   # grid-shape mismatch
            drv._ocean_mask_for_verify(on, grid9)

        class _AllLand(_FakeDriver):                          # no ocean => fail loud
            def static_land_fraction(self):
                return jnp.ones((2, 2))
        md.ModelDriver = _AllLand
        with pytest.raises(SystemExit, match="masks out EVERY column"):
            drv._ocean_mask_for_verify(on, grid4)
    finally:
        md.ModelDriver = orig


def test_ocean_mask_for_verify_all_ocean_is_a_flagged_noop(tmp_path, capsys):
    """A flat (no-land-mask) --base-config selects EVERY column under --ocean-only — a silent
    no-op that would report the GLOBAL bias as if ocean-only. It must be FLAGGED (mirrors the
    iter-477 preflight NOTE), not pass silently."""
    from types import SimpleNamespace

    import legoesm.driver.model_driver as md

    cfg_path = str(tmp_path / "flat.json")
    with open(cfg_path, "w") as f:
        json.dump({}, f)

    class _FlatDriver:
        def __init__(self, _cfg):
            pass

        def static_land_fraction(self):
            return jnp.zeros((2, 2))                  # flat => all ocean (no land mask)

    grid4 = SimpleNamespace(grid_area=jnp.ones(4))
    on = SimpleNamespace(ocean_only=True, base_config=cfg_path, max_land_fraction=0.5)
    orig = md.ModelDriver
    md.ModelDriver = _FlatDriver
    try:
        mask = drv._ocean_mask_for_verify(on, grid4)
    finally:
        md.ModelDriver = orig
    assert mask is not None and int(jnp.sum(mask)) == 4   # all 4 selected (no-op)
    out = capsys.readouterr().out
    assert "selected ALL 4 columns" in out and "no-op" in out


def test_ocean_only_flags_parse_defaults():
    a = drv._build_arg_parser().parse_args(
        ["--restart", "r", "--grid-type", "latlon", "--resolution", "8",
         "--nlev", "10", "--era5-zarr", "z"])
    assert a.ocean_only is False and a.max_land_fraction == 0.5 and a.base_config == ""
    a2 = drv._build_arg_parser().parse_args(
        ["--restart", "r", "--grid-type", "latlon", "--resolution", "8", "--nlev", "10",
         "--era5-zarr", "z", "--ocean-only", "--base-config", "c.json",
         "--max-land-fraction", "0.0"])
    assert a2.ocean_only is True and a2.base_config == "c.json" and a2.max_land_fraction == 0.0
