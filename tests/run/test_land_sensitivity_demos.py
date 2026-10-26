"""Direct tests for the four land sensitivity demo drivers.

Each test runs its demo end-to-end at a tiny size (a day or two of spin-up and
window) and asserts the one thing the demos exist to guarantee: every reverse-
mode sensitivity they report agrees with a central finite difference of the
same forward model.  Physical-bound and sign checks catch a gradient that is
finite but wrong in a way a finite difference at one point would not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5

from legoesm import constants
from scripts.run.land_sensitivity import column_setup as cs
from scripts.run.land_sensitivity import demo_adjoint_memory as adj
from scripts.run.land_sensitivity import demo_global_param_map as gm
from scripts.run.land_sensitivity import demo_process_jacobian as pj

jax.config.update("jax_enable_x64", True)


@pytest.mark.slow   # ~3-4 min: the noon Jacobian + FD checks compile the canopy step under AD
def test_process_jacobian_tiny(tmp_path):
    args = pj.build_parser().parse_args([
        "--spinup-days", "1", "--drydown-days", "2", "--fd-days", "0.5", "1.5",
        "--out-dir", str(tmp_path)])
    res, meta, checks, _ = pj.run_experiment(args)
    assert res.jac_per_theta.shape == (2, 5, 8) and res.jac_per_water.shape == (2, 5, 8)
    assert res.jac_column_water.shape == (2, 5) and np.all(np.isfinite(res.jac_column_water))
    # whole-column response = sum over layers of (per-layer response x that layer's
    # share of the column water): the chain rule through a uniform scaling
    share = res.theta * meta["dz_m"] * constants.rho_water / res.column_water[:, None]
    np.testing.assert_allclose(res.jac_column_water,
                               np.einsum("mfk,mk->mf", res.jac_per_water, share), rtol=1e-8)
    assert np.all(np.isfinite(res.jac_per_theta)) and np.all(np.isfinite(res.fluxes))
    # AD == FD for the Jacobian entries, both elasticities, dLE/dT and the
    # transpiration / soil-evaporation humidity derivatives at both noons.
    assert len(checks) == 2 * (2 * 3 + 3 + 2)
    assert res.jac_humidity_scale.shape == (2, 5)
    np.testing.assert_allclose(res.jac_humidity_scale[:, 0] + res.jac_humidity_scale[:, 1],
                               res.jac_humidity_scale[:, 4], rtol=1e-8)
    cs.require_all_pass(checks)
    # More soil water never reduces the transpiration demand (root uptake is monotone).
    assert np.all(res.jac_per_theta[:, 0, :] >= -1e-9)
    assert np.all(res.elasticity_water >= -1e-9)
    # per-kg/m2 scaling is the per-theta Jacobian divided by rho_w dz.
    np.testing.assert_allclose(
        res.jac_per_water[0, 0], res.jac_per_theta[0, 0] / (constants.rho_water * meta["dz_m"]))
    pj.plot(res, meta, tmp_path / "fig.png", texture="loam")
    assert (tmp_path / "fig.png").exists()


def test_noon_indices_requires_a_noon_step():
    axis = cs.make_time_axis(4, 5.0 * 3600.0, start_doy=1.0)   # 0, 5, 10, 15 h: no noon
    with pytest.raises(ValueError, match="noon"):
        pj.noon_indices(axis)


def test_season_precip_dry_spell_and_validation():
    axis = cs.make_time_axis(72, 3600.0, start_doy=1.0)
    p = np.asarray(adj.season_precip(axis, 2e-5, (1.0, 2.0)))
    assert np.all(p[:24] == 2e-5) and np.all(p[24:48] == 0.0) and np.all(p[48:] == 2e-5)
    with pytest.raises(ValueError):
        adj.season_precip(axis, 2e-5, (2.0, 1.0))


@pytest.mark.slow   # ~2-3 min: two season adjoints + FD checks
def test_adjoint_memory_tiny(tmp_path):
    args = adj.build_parser().parse_args([
        "--textures", "sand", "--spinup-days", "1", "--season-days", "2",
        "--dry-spell", "0.5", "1.0", "--fd-days", "0.25", "1.75",
        "--out-dir", str(tmp_path)])
    kernels, checks, meta = adj.run_experiment(args)
    assert [k.target for k in kernels] == ["water", "gpp"]
    cs.require_all_pass(checks)
    water = kernels[0]
    assert water.rain_kernel.shape == (48,) and water.rain_kernel_daily.shape == (2,)
    # An extra kg/m2 of rain cannot leave more than one kg/m2 in the column.
    assert np.all(water.rain_kernel <= 1.0 + 1e-6)
    # Rain in the last step is still (partly) in the column at the end.
    assert water.rain_kernel[-1] > 0.0
    adj.plot(kernels, meta, tmp_path / "fig.png", dry_spell=(0.5, 1.0))
    assert (tmp_path / "fig.png").exists()


_LAT_DEG = np.array([45.0, 10.0, 25.0, -75.0])   # forest, C4 grass, bare desert, glacier


def _synthetic_gsd(ncol=4, nlayer=8):
    """Four columns: temperate deciduous forest, C4 grass, bare soil, glacier."""
    npft = N_PFT_CLM5
    pft = np.zeros((1, ncol, npft))
    dbf = CLM5_PFT_NAMES.index("broadleaf_deciduous_temperate")
    c4 = CLM5_PFT_NAMES.index("c4_grass")
    pft[0, 0, dbf] = 1.0
    pft[0, 1, c4] = 1.0
    pft[0, 2, 0] = 1.0            # bare soil (land)
    pft[0, 3, 0] = 1.0            # glacier column (excluded by the glacier mask)
    lai = np.zeros((12, ncol, npft))
    lai[:, 0, dbf] = 4.0
    lai[:, 1, c4] = 1.5
    htop = np.zeros((12, ncol, npft))
    htop[:, 0, dbf] = 20.0
    htop[:, 1, c4] = 0.5
    sand = np.full((ncol, nlayer), 0.40)
    clay = np.full((ncol, nlayer), 0.20)
    sand[1], clay[1] = 0.80, 0.08
    z = np.zeros((ncol, nlayer))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(sand), clay_frac=jnp.asarray(clay),
        organic=jnp.asarray(z), bulk_density=jnp.asarray(z),
        soil_color=jnp.asarray(np.array([5, 12, 3, 18])), cell_area=jnp.ones(ncol),
        years=jnp.asarray([2015.0]),
        f_land=jnp.asarray([[1.0, 1.0, 1.0, 0.0]]), f_lake=jnp.zeros((1, ncol)),
        f_glacier=jnp.asarray([[0.0, 0.0, 0.0, 1.0]]), pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.zeros((12, ncol, npft)),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.zeros((12, ncol, npft)),
        config=GlobalSurfaceDataConfig(),
    )


def test_global_setup_selects_land_and_subsets_columns():
    gsd = _synthetic_gsd()
    lat, lon = np.deg2rad(_LAT_DEG), np.zeros(4)
    setup = gm.setup_from_surface_data(
        gsd, lat, lon, gm.base_config(), 180.0, land_frac_min=0.5, grid_shape=(4, 1))
    assert setup.idx.tolist() == [0, 1, 2]               # glacier column excluded
    assert setup.params.LAI.shape == (3,) and float(setup.params.LAI[0]) == 4.0
    assert float(setup.params.LAI[2]) == 0.0             # bare column
    assert setup.config.hydraulics.theta_sat.shape == (3, 8)
    assert np.all(setup.theta_fc > setup.theta_wp)
    # sandier column 1 holds less plant-available water
    assert setup.theta_fc[1] < setup.theta_fc[0]
    sub = gm.setup_from_surface_data(
        gsd, lat, lon, gm.base_config(), 180.0, land_frac_min=0.5, grid_shape=(4, 1),
        max_cells=1)
    assert sub.idx.size == 1 and sub.n_land_total == 3
    with pytest.raises(ValueError, match="Unknown precip profile"):
        gm.precip_per_column(setup.lat_rad, 1e-5, "monsoon")
    assert float(gm.zonal_precip_shape(0.0)) > float(gm.zonal_precip_shape(np.deg2rad(25.0)))
    mat, amp = gm.latitudinal_climate(np.deg2rad(np.array([0.0, 45.0, -45.0])))
    assert float(mat[0]) > float(mat[1]) and float(amp[1]) > 0.0 > float(amp[2])


def test_subset_mismatch_reporter_fails_only_on_real_mismatch(capsys):
    full = np.array([100.0, 50.0, 10.0])
    sel = np.array([3, 7, 9])
    gm._report_subset_mismatch("x", full.copy(), full, sel)            # identical: silent
    assert capsys.readouterr().out == ""
    gm._report_subset_mismatch("x", full * (1 + 1e-5), full, sel)      # round-off: reported
    assert "round-off" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="interact"):                # O(1): a defect
        gm._report_subset_mismatch("x", full * 1.05, full, sel)


def test_global_maps_round_trip_through_npz(tmp_path):
    n = 5
    rng = np.random.default_rng(0)
    setup = gm.GlobalSetup(lat_rad=jnp.zeros(n), lon_rad=jnp.zeros(n), idx=np.arange(n) * 3,
                           grid_shape=(3, 5), config=None, params=None, theta_wp=None,
                           theta_fc=None, t_init_k=None, n_land_total=n)
    grad = {t: {k: rng.random(n) for k in gm.INPUTS} for t in gm.TARGETS}
    maps = gm.SensitivityMaps(et=rng.random(n), gpp=rng.random(n), grad=grad,
                              percent_per_10pct=grad, converged=np.array([1, 1, 0, 1, 1]) > 0)
    gm._write_arrays(tmp_path, setup, maps)
    back = gm.load_maps(tmp_path / "global_param_map.npz", setup)
    np.testing.assert_array_equal(back.et, maps.et)
    np.testing.assert_array_equal(back.converged, maps.converged)
    for t in gm.TARGETS:
        for k in gm.INPUTS:
            np.testing.assert_array_equal(back.grad[t][k], maps.grad[t][k])
    with pytest.raises(ValueError, match="differ"):
        gm.load_maps(tmp_path / "global_param_map.npz", setup._replace(idx=np.arange(n)))


def test_unconverged_columns_flags_max_iters_exactly():
    n_iters = np.array([3, 49, 50, 71])
    np.testing.assert_array_equal(gm.unconverged_columns(n_iters, 50), [False, False, True, True])
    np.testing.assert_array_equal(gm.unconverged_columns(jnp.asarray(n_iters), 100),
                                  [False, False, False, False])


@pytest.mark.slow   # ~2-3 min: spin-up + one checkpointed reverse pass + two FD forward runs
def test_global_map_tiny_reverse_pass_matches_finite_difference(tmp_path):
    gsd = _synthetic_gsd()
    lat, lon = np.deg2rad(_LAT_DEG), np.zeros(4)
    setup = gm.setup_from_surface_data(
        gsd, lat, lon, gm.base_config(), 180.0, land_frac_min=0.5, grid_shape=(4, 1))
    precip_col = gm.precip_per_column(setup.lat_rad, 2e-5, "zonal")
    maps, state_spun, loss, timings = gm.compute_maps(
        setup, dt_s=10800.0, spinup_days=1, window_days=1, start_doy=180.0,
        precip_col=precip_col)
    assert set(maps.grad) == set(gm.TARGETS)
    assert all(set(g) == set(gm.INPUTS) for g in maps.grad.values())
    assert all(v.shape == (3,) for g in maps.grad.values() for v in g.values())
    assert maps.converged.shape == (3,) and maps.converged.dtype == bool
    # rain / root-depth sensitivities are defined everywhere; leaf-area / Vcmax
    # sensitivities only where there is vegetation (bare column -> NaN, not 0).
    for t in gm.TARGETS:
        for k in ("precip", "root_depth"):
            assert np.all(np.isfinite(maps.grad[t][k]))
        for k in ("lai", "vcmax"):
            assert np.all(np.isfinite(maps.grad[t][k][:2])) and np.isnan(maps.grad[t][k][2])
    assert np.all(maps.et > 0.0)
    # bare soil has (numerically) no GPP -> its GPP % entries are masked, not
    # divided by a ~1e-6 residual
    assert abs(maps.gpp[2]) < gm.GPP_MIN_FOR_PERCENT
    assert all(np.isnan(maps.percent_per_10pct["gpp"][k][2]) for k in gm.INPUTS)
    # chunked columns + a spun-state cache: same maps (columns are independent;
    # batch-layout round-off only), and the cache is reused on the second call
    cache = tmp_path / "spun.npz"
    key = {"resolution": 4}
    maps2, state2, _, _ = gm.compute_maps(
        setup, dt_s=10800.0, spinup_days=1, window_days=1, start_doy=180.0,
        precip_col=precip_col, column_chunks=2, spun_state_cache=cache, cache_key=key)
    assert cache.exists()
    np.testing.assert_allclose(maps2.et, maps.et, rtol=1e-6)
    for t in gm.TARGETS:
        for k in gm.INPUTS:
            np.testing.assert_allclose(maps2.grad[t][k], maps.grad[t][k], rtol=1e-5, atol=1e-9)
    maps3, state3, _, timings3 = gm.compute_maps(
        setup, dt_s=10800.0, spinup_days=1, window_days=1, start_doy=180.0,
        precip_col=precip_col, column_chunks=3, spun_state_cache=cache, cache_key=key)
    assert timings3["spinup_s"] < 1.0                       # loaded, not recomputed
    # same spun state -> same ET up to batch-layout round-off (3 chunks vs 2)
    np.testing.assert_allclose(maps3.et, maps2.et, rtol=1e-12)
    for a, b in zip(jax.tree_util.tree_leaves(state3), jax.tree_util.tree_leaves(state2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="different run"):
        gm.load_spun_state(cache, state2, {"resolution": 8})
    with pytest.raises(ValueError, match="column_chunks"):
        gm.compute_maps(setup, dt_s=10800.0, spinup_days=1, window_days=1, start_doy=180.0,
                        precip_col=precip_col, column_chunks=99)
    checks = gm.finite_difference_check(maps, loss, state_spun, inputs=["lai", "precip"],
                                        rtol=5e-3)
    assert len(checks) == 2 * len(gm.TARGETS)          # both targets per input
    assert any("dGPP_total" in c.label for c in checks)
    cs.require_all_pass(checks)
    # the subset path reproduces the full run's ET and gradient on the chosen
    # columns (asserted inside) and its FD checks pass too
    axis = cs.make_time_axis(8, 10800.0, start_doy=180.0)
    sub_checks = gm.finite_difference_check_subset(
        maps, setup, state_spun, time_axis=axis, dt_s=10800.0, precip_col=precip_col,
        max_cells=2, seed=0, inputs=["lai", "precip"], rtol=5e-3)
    assert len(sub_checks) == 4 and sub_checks[0].label.startswith("[subset of 2]")
    cs.require_all_pass(sub_checks)
    with pytest.raises(ValueError, match="fd-max-cells"):
        gm.finite_difference_check_subset(
            maps, setup, state_spun, time_axis=axis, dt_s=10800.0, precip_col=precip_col,
            max_cells=3, seed=0, inputs=["lai"], rtol=5e-3)
    assert checks[0].label.startswith("cell 0") or checks[0].label.startswith("cell 1")
    # more rain never lowers ET over a day
    assert np.all(maps.grad["et"]["precip"] >= -1e-9)
    # the figure honours the panel selection and rejects an unknown input
    # (the 4 columns are drawn on a 2 x 2 grid: the cell-edge helper needs >= 2
    # points per axis, which every real lat-lon grid has)
    setup_2x2 = setup._replace(grid_shape=(2, 2))
    gm.plot(maps, setup_2x2, np.array([45.0, 10.0]), np.array([0.0, 180.0]),
            tmp_path / "maps.png", window_days=1,
            panels={"et": ("vcmax", "precip"), "gpp": gm.INPUTS})
    assert (tmp_path / "maps.png").exists()
    with pytest.raises(ValueError, match="panel"):
        gm.plot(maps, setup_2x2, np.array([45.0, 10.0]), np.array([0.0, 180.0]),
                tmp_path / "bad.png", window_days=1, panels={"et": ("nope",)})
    full = gm.to_grid(maps.et, setup.idx, (4, 1))
    assert full.shape == (4, 1) and np.isnan(full[3, 0]) and np.isfinite(full[0, 0])


# --- process matrix ------------------------------------------------------------
from scripts.run.land_sensitivity import demo_process_matrix as pm  # noqa: E402


def test_process_matrix_log_factors_scale_config_and_canopy_leaves():
    config = cs.make_column_config("loam")
    params = cs.make_canopy_params(1)
    lf = jnp.zeros(len(pm.PARAMS))
    keys = [p.key for p in pm.PARAMS]
    lf = lf.at[keys.index("hydraulics.K_sat")].set(np.log(2.0))
    lf = lf.at[keys.index("params.LAI")].set(np.log(0.5))
    cfg, prm = pm.apply_log_factors(config, params, lf)
    np.testing.assert_allclose(float(cfg.hydraulics.K_sat), 2.0 * float(config.hydraulics.K_sat))
    np.testing.assert_allclose(np.asarray(prm.LAI), 0.5 * np.asarray(params.LAI))
    # untouched leaves are unchanged
    assert float(cfg.hydraulics.b_ch) == float(config.hydraulics.b_ch)
    np.testing.assert_array_equal(np.asarray(prm.Vcmax25_C3_leaf),
                                  np.asarray(params.Vcmax25_C3_leaf))
    # every row key resolves (a typo would raise here, not silently scale nothing)
    cfg2, prm2 = pm.apply_log_factors(config, params, jnp.full(len(pm.PARAMS), np.log(3.0)))
    assert float(cfg2.root_depth) == 3.0 * float(config.root_depth)
    assert float(cfg2.thermal.k_dry_coeff_a) == 3.0 * float(config.thermal.k_dry_coeff_a)


def test_process_matrix_percent_masks_tiny_window_totals():
    out = np.array([50.0, 0.5])                       # a real flux and a near-zero drainage
    jac = np.array([[5.0, -1.0], [0.4, 0.2]])
    pct = pm.percent_per_10pct(out, jac)
    np.testing.assert_allclose(pct[0], [1.0, -0.2])   # 10 * jac / 50
    assert np.all(np.isnan(pct[1]))                   # 0.5 < OUTPUT_MIN_FOR_PERCENT -> masked


def test_process_matrix_rejects_unknown_fd_param():
    mat = pm.ProcessMatrix("loam", np.ones(len(pm.OUTPUTS)),
                           np.zeros((len(pm.OUTPUTS), len(pm.PARAMS))),
                           np.zeros((len(pm.OUTPUTS), len(pm.PARAMS))))
    with pytest.raises(ValueError, match="unknown --fd-params"):
        pm.finite_difference_checks(mat, lambda x: x, param_keys=["params.nope"], rtol=1e-3)


@pytest.mark.slow   # ~3-5 min: one checkpointed reverse pass per output + FD checks
def test_process_matrix_tiny(tmp_path):
    args = pm.build_parser().parse_args([
        "--textures", "loam", "--spinup-days", "1", "--window-days", "1",
        "--fd-params", "params.LAI", "theta_fc", "--out-dir", str(tmp_path)])
    matrices, checks, meta = pm.run_experiment(args)
    (mat,) = matrices
    n_out, n_par = len(pm.OUTPUTS), len(pm.PARAMS)
    assert mat.jac_log.shape == (n_out, n_par) and mat.percent_per_10pct.shape == (n_out, n_par)
    assert np.all(np.isfinite(mat.jac_log)) and np.all(np.isfinite(mat.outputs))
    assert len(checks) == 2 * n_out
    cs.require_all_pass(checks)
    # the % entries are the log-Jacobian scaled by 10 / output where the window
    # total is at least one unit, NaN (masked) below (a one-day window leaves
    # some totals under that floor).
    k = pm.OUTPUTS.index("transpiration")
    defined = np.abs(mat.outputs) >= pm.OUTPUT_MIN_FOR_PERCENT
    assert defined[pm.OUTPUTS.index("stored soil water")]
    for i, ok in enumerate(defined):
        if ok:
            np.testing.assert_allclose(mat.percent_per_10pct[i],
                                       10.0 * mat.jac_log[i] / mat.outputs[i])
        else:
            assert np.all(np.isnan(mat.percent_per_10pct[i]))
    # a wetter stress threshold (higher field capacity) never raises transpiration
    j = [p.key for p in pm.PARAMS].index("theta_fc")
    assert mat.jac_log[k, j] <= 1e-9
    pm.plot(matrices, tmp_path / "fig.png", window_days=1)
    assert (tmp_path / "fig.png").exists()


# --- ET dry-down sensitivity from saved Jacobians --------------------------------
def _saved_drydown(n_days=2, seed=0):
    """Synthetic saved dry-down arrays shaped exactly like process_jacobian.npz/.json,
    internally consistent: transpiration + soil evaporation = ET for fluxes and
    derivatives, and the elasticities match the per-layer / per-flux arrays."""
    config = cs.make_column_config("loam")
    dz, _, z_iface = cs.soil_layers(config)
    n_layers = dz.size
    rng = np.random.default_rng(seed)
    theta = np.full((n_days, n_layers), 0.2)
    theta[:, 0] = 0.1                                 # top layer below the wilting point
    jac = np.zeros((n_days, 5, n_layers))
    jac[:, 0, :] = 60.0 * rng.random((n_days, n_layers))     # transpiration
    jac[:, 1, :] = 40.0 * rng.random((n_days, n_layers))     # soil evaporation
    jac[:, 4, :] = jac[:, 0, :] + jac[:, 1, :]               # ET
    fluxes = np.zeros((n_days, 5))
    fluxes[:, 0], fluxes[:, 1] = 30.0, 20.0
    fluxes[:, 4] = 50.0
    jsq = np.zeros((n_days, 5))
    jsq[:, 0], jsq[:, 1] = -4.0, -6.0
    jsq[:, 4] = jsq[:, 0] + jsq[:, 1]
    saved = {
        "day": np.arange(1.0, n_days + 1.0), "theta": theta, "fluxes": fluxes,
        "jac_per_theta": jac, "jac_humidity_scale": jsq,
        "elasticity_water": np.einsum("mk,mk->m", jac[:, 4], theta) / fluxes[:, 4],
        "elasticity_humidity": jsq[:, 4] / fluxes[:, 4],
        "dz_m": dz, "z_interface_m": z_iface,
    }
    saved_json = {
        "args": {"texture": "loam", "lat_deg": 40.0, "mean_annual_t_k": 285.0,
                 "seasonal_amp_k": 10.0, "dt_s": 3600.0, "drydown_days": n_days,
                 "start_doy": 152.0},
        "theta_wp": float(config.theta_wp),
        "gradient_checks": [cs.GradientCheck("synthetic", 1.0, 1.0, True, True)._asdict()],
    }
    return saved, saved_json, config


def test_layer_fraction_within_depth():
    np.testing.assert_allclose(pj.layer_fraction_within_depth([0.0, 0.5, 1.5, 3.0], 1.0),
                               [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="increase"):
        pj.layer_fraction_within_depth([0.0, 1.0, 1.0], 0.5)


def test_et_drydown_sensitivity_soil_water_split_and_depth():
    saved, saved_json, config = _saved_drydown()
    depth = 0.3
    sens = pj.et_drydown_sensitivity(saved, saved_json, depth_m=depth)
    frac = pj.layer_fraction_within_depth(saved["z_interface_m"], depth)
    assert 0.0 < frac[3] < 1.0 and frac[4] == 0.0      # 30 cm cuts the 17.5-37.5 cm layer
    # an even volumetric increment dtheta over the top 30 cm adds rho_w*0.3*dtheta
    # kg/m2 and changes each flux by sum_k J_k frac_k dtheta
    for j, (_, k) in enumerate(pj.ET_PARTS):
        expected = saved["jac_per_theta"][:, k, :] @ frac / (constants.rho_water * depth)
        np.testing.assert_allclose(sens.d_flux_d_water[:, j], expected, rtol=1e-12)
    np.testing.assert_allclose(sens.d_flux_d_water[:, 1] + sens.d_flux_d_water[:, 2],
                               sens.d_flux_d_water[:, 0], rtol=1e-12)
    # deeper spreading dilutes a Jacobian that is concentrated near the surface
    shallow_only = dict(saved, jac_per_theta=saved["jac_per_theta"].copy())
    shallow_only["jac_per_theta"][:, :, 2:] = 0.0
    shallow_only["elasticity_water"] = (np.einsum("mk,mk->m", shallow_only["jac_per_theta"][:, 4],
                                                  saved["theta"]) / saved["fluxes"][:, 4])
    s30 = pj.et_drydown_sensitivity(shallow_only, saved_json, depth_m=0.3)
    s50 = pj.et_drydown_sensitivity(shallow_only, saved_json, depth_m=0.5)
    np.testing.assert_allclose(s50.d_flux_d_water, s30.d_flux_d_water * 0.3 / 0.5, rtol=1e-12)
    np.testing.assert_allclose(sens.theta_mean,
                               (saved["theta"] * saved["dz_m"] * frac).sum(1) / depth)
    assert sens.theta_wp == config.theta_wp and sens.theta_fc == config.theta_fc


def test_et_drydown_sensitivity_vpd_split_matches_finite_difference_of_vpd():
    saved, saved_json, _ = _saved_drydown()
    sens = pj.et_drydown_sensitivity(saved, saved_json, depth_m=0.5)
    _, t_air, q_air, p = pj.noon_forcing_along_drydown(saved_json["args"])
    # independent route: central difference of VPD under the same humidity scaling
    from legoesm.thermo import saturation_vapor_pressure_aerk, vapor_pressure_from_specific_humidity

    def vpd(scale):
        return (np.asarray(saturation_vapor_pressure_aerk(jnp.asarray(t_air)))
                - np.asarray(vapor_pressure_from_specific_humidity(jnp.asarray(q_air * scale),
                                                                    jnp.asarray(p))))
    h = 1e-6
    dvpd_ds = (vpd(1.0 + h) - vpd(1.0 - h)) / (2.0 * h)
    for j, (_, k) in enumerate(pj.ET_PARTS):
        np.testing.assert_allclose(sens.d_flux_d_vpd[:, j],
                                   saved["jac_humidity_scale"][:, k] / dvpd_ds * pj.PA_PER_KPA,
                                   rtol=1e-6)
    # moistening the air lowers every flux here, so drier air (higher VPD) raises them
    assert np.all(sens.d_flux_d_vpd > 0.0)
    np.testing.assert_allclose(sens.vpd_kpa, vpd(1.0) / pj.PA_PER_KPA)


def test_et_drydown_sensitivity_refuses_inconsistent_saved_data():
    saved, saved_json, _ = _saved_drydown()
    ok = dict(depth_m=0.3)
    with pytest.raises(RuntimeError, match="water elasticity"):
        pj.et_drydown_sensitivity(dict(saved, elasticity_water=saved["elasticity_water"] * 1.01),
                                  saved_json, **ok)
    with pytest.raises(RuntimeError, match="humidity elasticity"):
        pj.et_drydown_sensitivity(
            dict(saved, elasticity_humidity=saved["elasticity_humidity"] * 1.01), saved_json, **ok)
    bad_split = dict(saved, fluxes=saved["fluxes"].copy())
    bad_split["fluxes"][:, 1] += 1.0                    # soil evaporation no longer sums to ET
    with pytest.raises(RuntimeError, match="transpiration \\+ soil evaporation vs ET"):
        pj.et_drydown_sensitivity(bad_split, saved_json, **ok)
    failed = dict(saved_json, gradient_checks=[
        cs.GradientCheck("synthetic", 1.0, 2.0, False, True)._asdict()])
    with pytest.raises(RuntimeError, match="FAILED"):
        pj.et_drydown_sensitivity(saved, failed, **ok)
    with pytest.raises(ValueError, match="day axis"):
        pj.et_drydown_sensitivity(dict(saved, day=saved["day"] + 1.0), saved_json, **ok)
    old = {k: v for k, v in saved.items() if k != "jac_humidity_scale"}
    with pytest.raises(ValueError, match="rerun"):
        pj.et_drydown_sensitivity(old, saved_json, **ok)
    with pytest.raises(ValueError, match="depth_m"):
        pj.et_drydown_sensitivity(saved, saved_json, depth_m=0.0)


def test_et_sensitivity_from_saved_cli_writes_one_figure_per_depth(tmp_path):
    import json
    saved, saved_json, _ = _saved_drydown()
    np.savez(tmp_path / "process_jacobian.npz", **saved)
    (tmp_path / "process_jacobian.json").write_text(json.dumps(saved_json))
    assert pj.main(["--et-sensitivity-from-saved", "--out-dir", str(tmp_path)]) == 0
    for depth_cm in pj.DEFAULT_WATER_DEPTHS_CM:
        assert (tmp_path / pj.et_sensitivity_filename(depth_cm * pj.M_PER_CM)).exists()
    assert pj.et_sensitivity_filename(0.3) == "et_drydown_sensitivity_30cm.png"


# --- rain-pulse memory in transpiration and soil evaporation --------------------------
from scripts.run.land_sensitivity import demo_pulse_memory as pulse  # noqa: E402


def test_pulse_precip_puts_the_pulse_in_the_first_hours():
    axis = cs.make_time_axis(48, 3600.0, start_doy=152.0)
    rate, n_pulse = pulse.pulse_precip(axis, pulse_mm=20.0, pulse_hours=6.0, dt_s=3600.0)
    rate = np.asarray(rate)
    assert n_pulse == 6
    np.testing.assert_allclose(rate.sum() * 3600.0, 20.0, rtol=1e-12)
    assert np.all(rate[:6] > 0.0) and np.all(rate[6:] == 0.0)
    for bad in (dict(pulse_hours=1.5, dt_s=7200.0), dict(pulse_hours=0.0, dt_s=3600.0),
                dict(pulse_hours=49.0, dt_s=3600.0)):
        with pytest.raises(ValueError, match="pulse_hours"):
            pulse.pulse_precip(axis, pulse_mm=20.0, **bad)
    with pytest.raises(ValueError, match="pulse_mm"):
        pulse.pulse_precip(axis, pulse_mm=-1.0, pulse_hours=6.0, dt_s=3600.0)


def test_memory_targets_include_every_water_fate():
    config = cs.make_column_config("loam")
    dz, _, _ = cs.soil_layers(config)
    fns = adj.make_target_fns(config, cs.make_canopy_params(1), lat_rad=0.7,
                              time_axis=cs.make_time_axis(2, 3600.0, start_doy=152.0),
                              dt_s=3600.0, dz=dz, state0=cs.initial_state(config, t_init_k=290.0),
                              remat=False)
    assert set(pulse.FATES) <= set(fns) and set(adj.DEFAULT_TARGETS) <= set(fns)
    assert set(adj.TARGETS) == set(fns) == set(adj.TARGET_UNITS)


def test_water_fate_totals_and_kernel_assembly():
    config = cs.make_column_config("loam")
    dz, _, _ = cs.soil_layers(config)
    final = cs.initial_state(config, t_init_k=290.0)
    n = 48
    hist = cs.ColumnHistory(
        lhflx=jnp.zeros(n), shflx=jnp.zeros(n), le_canopy=jnp.full(n, 60.0),
        le_soil=jnp.full(n, 30.0), gpp=jnp.zeros(n), t_sfc=jnp.zeros(n),
        runoff=jnp.full(n, 1e-6), theta=jnp.zeros((n, dz.size)), n_iters=jnp.zeros(n))
    tot = np.asarray(adj.water_fate_totals(final, hist, dz, 3600.0))
    np.testing.assert_allclose(tot[:3], [60.0 * n * 3600.0 / constants.L_v,
                                         30.0 * n * 3600.0 / constants.L_v, 1e-6 * n * 3600.0])
    np.testing.assert_allclose(tot[3], float(cs.column_storage_kg_m2(final, dz)))
    assert adj.WATER_FATES == pulse.FATES and len(tot) == len(adj.WATER_FATES)
    g_p = np.arange(n, dtype=float)
    kern = adj.kernels_from_gradients("loam", "storage", 1.0, g_p, np.ones(n), 3600.0, 24)
    np.testing.assert_allclose(kern.rain_kernel, g_p / 3600.0)
    np.testing.assert_allclose(kern.rain_kernel_daily, [g_p[:24].mean() / 3600.0,
                                                        g_p[24:].mean() / 3600.0])
    np.testing.assert_allclose(kern.t_air_kernel_daily, [24.0, 24.0])
    with pytest.raises(FloatingPointError, match="non-finite"):
        adj.kernels_from_gradients("loam", "storage", float("nan"), g_p, g_p, 3600.0, 24)


def test_require_partition_raises_only_when_fates_do_not_add_to_one():
    ok = pulse.PulseMemory("loam", np.zeros(1), {}, {}, {}, {}, 1e-6, 0.0, {})
    pulse.require_partition([ok])
    with pytest.raises(RuntimeError, match="add to one"):
        pulse.require_partition([ok, ok._replace(texture="clay", partition_error=0.2)])


@pytest.mark.slow   # ~2-4 min: one four-fate reverse-mode Jacobian on a 2-day window
def test_pulse_memory_tiny(tmp_path):
    args = pulse.build_parser().parse_args([
        "--spinup-days", "1", "--window-days", "2",
        "--pulse-mm", "10", "--pulse-hours", "3", "--out-dir", str(tmp_path)])
    assert args.textures == ["loam"] and args.fd_days == []
    results, checks, meta = pulse.run_experiment(args)
    assert checks == []                                  # finite differences are opt-in
    (r,) = results
    pulse.require_partition(results)
    assert abs(r.budget_residual) < pulse.BUDGET_ATOL_KG_M2
    np.testing.assert_allclose(sum(r.pulse_fate.values()), 1.0, atol=pulse.PARTITION_ATOL)
    # the identity is not vacuous: dropping a fate breaks it
    without_storage = sum(r.kernels[f].rain_kernel for f in pulse.FATES if f != "storage")
    assert np.max(np.abs(without_storage - 1.0)) > 0.1
    for fate in pulse.FATES:
        assert r.kernel_daily[fate].shape == (2,)
        assert np.all(np.isfinite(r.kernels[fate].rain_kernel))
    assert r.pulse_fate["transpiration"] >= 0.0 and r.pulse_fate["soil_evaporation"] >= 0.0
    assert meta["n_pulse"] == 3
    pulse.plot(results, tmp_path / "fig.png", pulse_mm=10.0, pulse_hours=3.0)
    assert (tmp_path / "fig.png").exists()


def _saved_pulse_run(out_dir, n_days=3, dt_s=3600.0):
    """Files shaped like a pulse_memory run's npz/json, internally consistent."""
    import json
    spd = int(86400.0 / dt_s)
    rng = np.random.default_rng(1)
    arrays, window_value = {}, {}
    for i, fate in enumerate(pulse.FATES):
        arrays[f"loam_{fate}_rain_kernel"] = 0.25 + 0.01 * rng.random(n_days * spd)
        arrays[f"loam_{fate}_t_air_kernel"] = rng.random(n_days * spd)
        window_value[fate] = float(i)
    for proc in pulse.PROCESSES:
        arrays[f"loam_{proc}_base_daily_mm"] = rng.random(n_days)
    np.savez(out_dir / "pulse_memory.npz", **arrays)
    (out_dir / "pulse_memory.json").write_text(json.dumps({
        "args": {"dt_s": dt_s, "pulse_mm": 20.0, "pulse_hours": 6.0, "textures": "['loam']"},
        "summary": {"loam": {"pulse_fate": {f: 0.25 for f in pulse.FATES},
                             "window_value": window_value, "partition_error": 1e-5,
                             "budget_residual_kg_m2": 1e-6}}}))
    return arrays


def test_pulse_memory_replot_from_saved(tmp_path):
    arrays = _saved_pulse_run(tmp_path)
    results, saved = pulse.load_results(tmp_path)
    (r,) = results
    assert r.texture == "loam" and r.day.shape == (3,)
    np.testing.assert_allclose(r.kernel_daily["soil_evaporation"],
                               arrays["loam_soil_evaporation_rain_kernel"].reshape(3, 24).mean(1))
    np.testing.assert_allclose(r.kernels["storage"].rain_kernel, arrays["loam_storage_rain_kernel"])
    assert r.window_value["runoff"] == 2.0
    pulse.plot(results, tmp_path / "t_e.png", pulse_mm=20.0, pulse_hours=6.0,
               kernel_fates=pulse.PROCESSES, max_day=2.0)
    assert (tmp_path / "t_e.png").exists()
    with pytest.raises(ValueError, match="kernel_fates"):
        pulse.plot(results, tmp_path / "bad.png", pulse_mm=20.0, pulse_hours=6.0,
                   kernel_fates=("evapotranspiration",))
    with pytest.raises(ValueError, match="max_day"):
        pulse.plot(results, tmp_path / "bad.png", pulse_mm=20.0, pulse_hours=6.0, max_day=0.0)
    args = pulse.build_parser().parse_args([])
    assert args.plot_fates == list(pulse.PROCESSES) and args.plot_days == 35.0
    (tmp_path / "pulse_memory.png").unlink(missing_ok=True)
    assert pulse.main(["--replot-from-saved", "--plot-days", "2", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "pulse_memory.png").exists()
