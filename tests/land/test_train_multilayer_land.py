"""Multilayer (Richards) land calibration core — synthetic data, no network."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from scripts.run.train_land_params_era5 import _N_PFT
from scripts.run.train_multilayer_land_era5 import (
    forward_ml, loss_ml, constrain_ext, init_ext_params, baked_init_params, BOUNDS_EXT,
    build_multilayer_cfg, _split_cells, json_init_params)


def _synthetic_data(ncol=12):
    rng = np.random.default_rng(0)
    lat = np.deg2rad(rng.uniform(-80, 80, ncol))
    pft = rng.random((ncol, _N_PFT)); pft /= pft.sum(1, keepdims=True)
    z = lambda v: jnp.full((4, ncol), v)         # (NH, ncol) subdaily forcing
    Tair = 285 - 0.3 * np.abs(np.rad2deg(lat))
    f = [AtmToSurface(
        sw_down=z(200.0), lw_down=z(320.0), precip_total=z(2e-5), precip_snow=z(0.0),
        T_lowest=jnp.broadcast_to(jnp.asarray(Tair), (4, ncol)), q_lowest=z(5e-3),
        u_lowest=z(3.0), v_lowest=z(2.0), p_lowest=z(9.9e4), p_surface=z(1.0e5),
        rho_lowest=z(1.2), cos_zenith=z(0.5), co2_ppmv=z(412.0),
        has_radiation=z(1.0), has_precipitation=z(1.0)) for _ in range(12)]
    dom = pft.argmax(1); oh = np.zeros((ncol, _N_PFT)); oh[np.arange(ncol), dom] = 1.0
    c = lambda v: jnp.full((ncol,), v)
    data = dict(forc=f, lat=jnp.asarray(lat), pft=jnp.asarray(pft),
                fg=jnp.asarray(rng.random(ncol) * 0.3), wp=c(0.12), fc=c(0.30),
                pct_sand=jnp.asarray(rng.uniform(20, 80, ncol)),   # %sand for texture k/C
                pct_clay=jnp.asarray(rng.uniform(5, 40, ncol)),
                skt=jnp.broadcast_to(jnp.asarray(Tair), (12, ncol)),
                alb=jnp.full((12, ncol), 0.2), t0=jnp.asarray(Tair),
                soil_albedo=jnp.asarray(rng.uniform(0.08, 0.30, ncol)),   # CLM soil-colour
                dom_onehot=jnp.asarray(oh), w=jnp.cos(jnp.asarray(lat)))
    # per-column van-Genuchten soil (loam-ish, physical)
    for k, v in dict(theta_r=0.05, theta_sat=0.43, alpha_vg=3.6, n_vg=1.56,
                     K_sat=2.9e-6).items():
        data["vg_" + k] = c(v)
    # soil-moisture target (annual-mean 0-28cm) + the model root-zone depth weights
    import scripts.run.train_multilayer_land_era5 as _M
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
    _gr = make_soil_grid(SoilGridConfig(n_layers=_M._N_LAYERS, total_depth=_M._SOIL_DEPTH_M,
                                        growth_factor=_M._SOIL_GROWTH))
    _dz = np.asarray(_gr.dz); _bot = np.cumsum(_dz); _ov = np.clip(0.28 - (_bot - _dz), 0.0, _dz)
    data["sm"] = c(0.25)
    data["rz_w"] = jnp.asarray(_ov[_ov > 1e-9])
    # latent-heat target (positive-up W/m2), the evaporation leg of the dual target
    data["le"] = jnp.full((12, ncol), 40.0)
    return data


def test_constrain_bounds():
    cp = constrain_ext(init_ext_params())
    for k, (lo, hi) in BOUNDS_EXT.items():
        v = np.asarray(cp[k])
        assert np.all(v >= lo - 1e-9) and np.all(v <= hi + 1e-9)


def test_baked_warm_start_roundtrips_to_production_params():
    """--init-from baked must START the tune AT the production baked multilayer params
    (so the re-tune REFINES them, not climbs from the CLM5 prior).  constrain(baked_init)
    must reproduce the _TUNED_*_MULTILAYER values and stay in bounds."""
    from legoesm.land import clm_surface_map as C
    cp = constrain_ext(baked_init_params())
    # in-bounds like any param set
    for k, (lo, hi) in BOUNDS_EXT.items():
        v = np.asarray(cp[k])
        assert np.all(v >= lo - 1e-9) and np.all(v <= hi + 1e-9), k
    # round-trips the baked constrained values (within the _inv_ext clip epsilon)
    np.testing.assert_allclose(cp["pft_alb"], C._TUNED_PFT_ALBEDO_MULTILAYER, atol=2e-6)
    np.testing.assert_allclose(cp["pft_z0"], C._TUNED_PFT_Z0_MULTILAYER, atol=2e-6)
    assert abs(float(cp["glac_alb"]) - C.TUNED_GLACIER_ALBEDO_MULTILAYER) < 2e-6
    assert abs(float(cp["snow_max"]) - C.TUNED_SNOW_ALBEDO_MAX_MULTILAYER) < 2e-6
    # differs from the CLM5 prior (it is a genuine warm start, not the default init)
    assert abs(float(cp["glac_alb"]) - float(constrain_ext(init_ext_params())["glac_alb"])) > 1e-3


def test_json_init_params_round_trips_a_saved_checkpoint(tmp_path):
    """--init-json must START the tune AT a saved CONSTRAINED tuned JSON (inverse of the
    constrain applied at save time), so a re-tune REFINES the current best.  A key absent
    from an older checkpoint keeps its baked default (layered), and forward() of the
    warm-started constrained params reproduces the saved values."""
    import json
    raw = baked_init_params()
    saved = {k: np.asarray(v).tolist() for k, v in constrain_ext(raw).items()}
    p = tmp_path / "tuned.json"
    p.write_text(json.dumps(saved))
    back = json_init_params(str(p))
    # every baked key present + constrain(json_init) reproduces the saved constrained JSON
    assert set(back) == set(raw)
    cp = constrain_ext(back)
    for k in saved:
        np.testing.assert_allclose(np.asarray(cp[k]), np.asarray(saved[k]), rtol=1e-4,
                                   atol=1e-4, err_msg=k)
    # layering: a JSON with only a subset of keys keeps baked for the rest
    subset = {k: saved[k] for k in list(saved)[:3]}
    p2 = tmp_path / "partial.json"
    p2.write_text(json.dumps(subset))
    back2 = json_init_params(str(p2))
    assert set(back2) == set(raw)
    missing = [k for k in raw if k not in subset]
    np.testing.assert_allclose(np.asarray(back2[missing[0]]), np.asarray(raw[missing[0]]),
                               rtol=1e-6, atol=1e-6)


def test_forward_finite_and_physical():
    data = _synthetic_data()
    T, A, W, L = forward_ml(constrain_ext(init_ext_params()), data)
    assert T.shape == (12, 12) and A.shape == (12, 12)
    assert jnp.all(jnp.isfinite(T)) and jnp.all(jnp.isfinite(A))
    assert 230.0 < float(T.mean()) < 330.0       # no runaway / freeze-out
    assert jnp.all((A > 0.0) & (A < 1.0))
    # root-zone soil moisture is physical (within [theta_r, theta_sat]) and finite
    assert W.shape == (12,) and jnp.all(jnp.isfinite(W))
    assert jnp.all((W > 0.0) & (W < 0.6))
    # monthly latent heat is finite and physically bounded
    assert L.shape == (12, 12) and jnp.all(jnp.isfinite(L))
    assert -200.0 < float(L.mean()) < 400.0


def test_loss_is_differentiable():
    data = _synthetic_data()
    g = jax.grad(lambda p: loss_ml(p, data)[0])(init_ext_params())
    assert all(jnp.all(jnp.isfinite(v)) for v in g.values())
    # In the DEFAULT (MOST, stomata ON) config these knobs are trainable: albedo
    # (net SW), per-PFT soil thermal inertia (C_soil, k_solid -> seasonal cycle),
    # the MOST roughness z0 and the unfrozen Farquhar canopy-conductance params
    # (Vc_max25/g1/LCMA, constrained by the latent-heat leg of the dual target).
    # theta_wp is data-dependent (test_water_stress_response); Ch has no gradient
    # path under MOST and is frozen out by _inactive_keys in train().
    for k in ("pft_alb", "pft_kscale", "pft_cscale", "pft_z0",
              "pft_vcmax", "pft_g1", "pft_lcma"):
        assert float(jnp.max(jnp.abs(g[k]))) > 0.0, f"{k} has zero gradient"
    # the soil-moisture target makes the porosity scale (theta_sat) trainable
    assert float(jnp.max(jnp.abs(g["pft_smscale"]))) > 0.0, "pft_smscale has zero gradient"


def test_snow_albedo_params_trainable():
    """With snow present (cold air + snowfall) the exposed snow-albedo params carry a
    non-zero gradient -> they actually drive the surface albedo (snow cover fraction +
    snow brightness), not inert LandAlbedoConfig defaults."""
    data = dict(_synthetic_data())
    cold = 250.0                                   # below freezing -> snow accumulates
    data["skt"] = jnp.full_like(data["skt"], cold)
    data["t0"] = jnp.full_like(data["t0"], cold)
    data["forc"] = [f._replace(
        T_lowest=jnp.full_like(f.T_lowest, cold),
        precip_snow=jnp.full_like(f.precip_snow, 2e-5),
        precip_total=jnp.full_like(f.precip_total, 2e-5)) for f in data["forc"]]
    g = jax.grad(lambda p: loss_ml(p, data)[0])(init_ext_params())
    for k in ("snow_max", "snow_min", "snow_dcrit", "snow_tau_days"):
        assert jnp.all(jnp.isfinite(g[k])), f"{k} non-finite gradient"
    # With continuously-FRESH snow (snow_age~0) the fresh-snow albedo and the snow-cover
    # threshold drive the loss; snow_min / tau only engage once snow AGES (snow_age>0),
    # so they are legitimately inert here (both are non-zero on real 24-h data, where
    # snow ages between events).
    for k in ("snow_max", "snow_dcrit"):
        assert float(jnp.abs(g[k])) > 0.0, f"{k} inert (no fresh-snow-albedo gradient)"


def test_split_cells_disjoint_and_complete():
    """The 80/20 train/test split partitions the cells: disjoint and covering."""
    data = _synthetic_data(ncol=20)
    train, test = _split_cells(data, 0.2, seed=0)
    assert int(test["lat"].shape[0]) == 4 and int(train["lat"].shape[0]) == 16
    orig = set(np.asarray(data["lat"]).round(9).tolist())
    tr = set(np.asarray(train["lat"]).round(9).tolist())
    te = set(np.asarray(test["lat"]).round(9).tolist())
    assert tr.isdisjoint(te), "train/test overlap -> leakage"
    assert (tr | te) == orig, "split drops or duplicates cells"


def test_lam_sm_zero_is_true_noop():
    """lam_sm=0 must SKIP the soil-moisture term entirely (no 0*NaN poisoning): the
    aux smse is exactly 0 and the total loss equals the sum of the other terms."""
    data = _synthetic_data()
    p = init_ext_params()
    l, (tm, am, pp, sa, sm, gb, le) = loss_ml(p, data, lam_sm=0.0)
    assert float(sm) == 0.0
    # the SM term contributes nothing: loss == sum of the other weighted terms
    import scripts.run.train_multilayer_land_era5 as _M
    expect = (float(tm) + _M._LAM_ALB * float(am) + _M._LAM_PFT * float(pp)
              + _M._LAM_AMP * float(sa) + _M._LAM_LE * float(le))
    assert abs(float(l) - expect) < 1e-6


def test_lam_tbias_penalizes_global_bias():
    """--lam-tbias adds an area-weighted global skin-T BIAS penalty on top of the RMSE
    term: the aux exposes the signed bias (weight-independent), and a positive lam_tbias
    raises the loss by exactly lam_tbias * gbias**2, differentiably."""
    data = _synthetic_data()
    p = init_ext_params()
    l0, aux0 = loss_ml(p, data, lam_tbias=0.0)
    l1, aux1 = loss_ml(p, data, lam_tbias=50.0)
    gbias = float(aux0[5])                                   # signed global skin-T bias [K]
    assert float(aux1[5]) == gbias                           # diagnostic independent of the weight
    assert abs((float(l1) - float(l0)) - 50.0 * gbias ** 2) < 1e-5
    g = jax.grad(lambda q: loss_ml(q, data, lam_tbias=50.0)[0])(p)
    assert all(jnp.all(jnp.isfinite(v)) for v in g.values())


def test_soil_moisture_loss_is_nan_safe():
    """A non-finite soil-moisture target (or model) must NOT poison the loss/gradient —
    finite-masking drops those cells (codex: a diverged forward must not NaN the run)."""
    data = _synthetic_data()
    sm = np.asarray(data["sm"]).copy(); sm[0] = np.nan       # one missing target cell
    data = dict(data); data["sm"] = jnp.asarray(sm)
    g = jax.grad(lambda q: loss_ml(q, data)[0])(init_ext_params())
    assert all(jnp.all(jnp.isfinite(v)) for v in g.values()), "NaN SM target poisoned the gradient"


def test_all_nan_soil_moisture_target_is_fully_masked():
    """Backward-compat: an OLD npz without soil moisture yields an all-NaN SM target
    (_pack KeyError fallback).  The finite-mask drops every cell -> smse is exactly 0
    and the loss/gradient stay finite, so legacy inputs still train."""
    data = dict(_synthetic_data())
    data["sm"] = jnp.full_like(data["sm"], jnp.nan)
    l, aux = loss_ml(init_ext_params(), data)
    assert float(aux[4]) == 0.0 and jnp.isfinite(l)          # smse (index 4) masked to 0
    g = jax.grad(lambda q: loss_ml(q, data)[0])(init_ext_params())
    assert all(jnp.all(jnp.isfinite(v)) for v in g.values())


def test_latent_heat_loss_is_nan_safe_and_maskable():
    """One NaN LE target cell must not poison the gradient; an all-NaN LE target
    (legacy npz without slhf_wm2) masks the term to exactly 0 (backward compat)."""
    data = dict(_synthetic_data())
    le = np.asarray(data["le"]).copy(); le[:, 0] = np.nan
    data["le"] = jnp.asarray(le)
    g = jax.grad(lambda q: loss_ml(q, data)[0])(init_ext_params())
    assert all(jnp.all(jnp.isfinite(v)) for v in g.values()), "NaN LE target poisoned the gradient"
    data["le"] = jnp.full_like(data["le"], jnp.nan)
    l, aux = loss_ml(init_ext_params(), data)
    assert float(aux[6]) == 0.0 and jnp.isfinite(l)          # lemse (index 6) masked to 0


def test_inactive_keys_track_mode():
    """The strict no-inert-parameters rule freezes exactly the keys with no gradient
    path in each mode; the default mode (MOST + stomata + no elev bands) trains the
    Farquhar params and freezes only Ch + the elevation-band closures."""
    import scripts.run.train_multilayer_land_era5 as M
    saved = (M._BULK_SCHEME, M._STOMATA_ON, M._ELEV_BANDS_ON)
    try:
        M._BULK_SCHEME, M._STOMATA_ON, M._ELEV_BANDS_ON = "most", True, False
        assert M._inactive_keys() == {"pft_ch", "elev_lapse", "elev_sw_grad",
                                      "elev_lw_lapse", "glac_ice_alb", "snow_zenith"}
        M._BULK_SCHEME, M._STOMATA_ON, M._ELEV_BANDS_ON = "constant", False, True
        # snow_zenith frozen in every mode: no albedo call site passes cos_zenith
        assert M._inactive_keys() == {"pft_z0", "pft_vcmax", "pft_g1", "pft_lcma",
                                      "snow_zenith"}
    finally:
        M._BULK_SCHEME, M._STOMATA_ON, M._ELEV_BANDS_ON = saved


def test_porosity_scale_keeps_theta_sat_above_field_capacity():
    """The pft_smscale clamp must keep the scaled porosity above theta_r AND the plant
    field capacity for ANY in-bounds scale (incl. the 0.7 minimum) so van-Genuchten +
    btran stay well-posed (codex)."""
    data = _synthetic_data()
    p = init_ext_params()
    # force the porosity scale to its 0.7 minimum (raw -> -inf-ish sigmoid)
    p = dict(p); p["pft_smscale"] = jnp.full_like(p["pft_smscale"], -20.0)
    cp = constrain_ext(p)
    assert float(jnp.max(cp["pft_smscale"])) < 0.72            # at the 0.7 floor
    cfg, lp, hyd, _ = build_multilayer_cfg(cp, data)
    plant_fc = lp.theta_fc                                      # (ncol,) plant field capacity
    assert jnp.all(hyd.theta_sat[:, 0] >= hyd.theta_r[:, 0]), "theta_sat below theta_r"
    assert jnp.all(hyd.theta_sat[:, 0] >= plant_fc + 0.0199), "theta_sat below plant FC -> btran breaks"


def test_bulk_stomata_toggle_params():
    """--bulk constant makes Ch trainable (z0 inert); stomata ON (default) makes the
    Farquhar photosynthesis params (Vc_max25/g1/LCMA) trainable."""
    import scripts.run.train_multilayer_land_era5 as M
    data = _synthetic_data()
    saved = (M._BULK_SCHEME, M._STOMATA_ON)
    try:
        M._BULK_SCHEME, M._STOMATA_ON = "constant", False
        g = jax.grad(lambda p: M.loss_ml(p, data)[0])(init_ext_params())
        assert float(jnp.max(jnp.abs(g["pft_ch"]))) > 0.0, "pft_ch zero under constant bulk"
        M._BULK_SCHEME, M._STOMATA_ON = "most", True
        g = jax.grad(lambda p: M.loss_ml(p, data)[0])(init_ext_params())
        for k in ("pft_vcmax", "pft_g1", "pft_lcma"):
            assert float(jnp.max(jnp.abs(g[k]))) > 0.0, f"{k} zero under most+stomata"
    finally:
        M._BULK_SCHEME, M._STOMATA_ON = saved


def _one_step_lhflx(cp, data, theta_val, T_init=300.0):
    """Latent heat flux from ONE multilayer step with the soil moisture pinned in the
    stress band (theta_val).  Single-step avoids the moisture equilibrium where ET is
    pinned to precip by mass balance and so is insensitive to the plant params."""
    from scripts.run.train_multilayer_land_era5 import build_multilayer_cfg
    from legoesm.land.multilayer_land import (
        step_multilayer_land, init_multilayer_land_state)
    from legoesm.land.soil_hydraulics import psi_from_theta
    n = data["lat"].shape[0]
    cfg, lp, hyd, cs0 = build_multilayer_cfg(cp, data)
    st = init_multilayer_land_state(n, cfg, T_init=280.0)
    st = jax.tree.map(lambda x: x.astype(jnp.float64), st)
    th = jnp.full_like(st.theta_soil, theta_val)
    st = st._replace(T_soil=jnp.full_like(st.T_soil, T_init), theta_soil=th,
                     psi_soil=psi_from_theta(th, hyd))
    f = jax.tree.map(lambda x: x[2], data["forc"][6])   # noon hour, July -> (ncol,)
    _, r, _ = step_multilayer_land(st, f, cfg, 1.0, 3600.0, lat=data["lat"],
                                   doy=195.0, land_params=lp, carbon_state=cs0)
    return float(jnp.mean(r.lhflx))


def test_water_stress_response():
    """Raising the plant wilting point (earlier stress) cuts the latent heat flux —
    the calibratable btran response."""
    data = _synthetic_data(); base = constrain_ext(init_ext_params())
    hi_stress = dict(base, pft_wp=jnp.full(_N_PFT, 0.23), pft_fcgap=jnp.full(_N_PFT, 0.14))
    lo_stress = dict(base, pft_wp=jnp.full(_N_PFT, 0.05), pft_fcgap=jnp.full(_N_PFT, 0.14))
    # theta=0.18 sits between the two wilting points -> hi-wp cell is stressed
    assert _one_step_lhflx(hi_stress, data, 0.18) < _one_step_lhflx(lo_stress, data, 0.18) - 1.0


def test_adam_reduces_loss():
    data = _synthetic_data()
    p = init_ext_params()
    # jit: the eager (op-by-op) backward through the two-year scan segfaults with
    # the stomata-ON forward; the compiled backward is what train() runs anyway.
    vg = jax.jit(jax.value_and_grad(lambda q: loss_ml(q, data)[0]))
    opt = optax.adam(3e-2); s = opt.init(p)
    l0 = float(vg(p)[0])
    for _ in range(4):
        l, gr = vg(p); u, s = opt.update(gr, s); p = optax.apply_updates(p, u)
    assert float(l) < l0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
