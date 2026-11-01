"""CloudBench → LES-suite artifact conversion (les_suite/cloudbench.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from legoesm.atmosphere.les_suite import cloudbench as cb
from legoesm.atmosphere.les_suite.bridge import BridgeError, load_artifact, save_artifact
from legoesm.atmosphere.physics._shared import exner_function


def _grid(nx=6, ny=6, nz=12):
    return tuple(cb._CB_DOMAIN_M / (n - 1) for n in (nx, ny, nz))


def test_rhok_uniform_shear_neutral_is_the_smagorinsky_lilly_value():
    nx, ny, nz = 6, 6, 12
    dx, dy, dz = _grid(nx, ny, nz)
    a = 0.01
    z = np.arange(nz) * dz
    u = np.broadcast_to(a * z, (nx, ny, nz)).copy()
    zero = np.zeros_like(u)
    T = np.full_like(u, 290.0)
    p = np.full_like(u, 9.0e4)                      # uniform p: theta_v uniform, Ri = 0
    rho = np.full_like(u, 1.1)
    rk = cb.smagorinsky_lilly_rhok(u, zero, zero, T, zero, zero, p, rho, dx, dy, dz)
    delta2 = (dx * dy * dz) ** (2.0 / 3.0)
    want = 1.1 * ((cb._CB_C_S ** 2 / cb._CB_PR_T) * delta2 * a + cb._CB_D_MOL)
    assert np.allclose(rk[..., 1:-1], want, rtol=1e-12, atol=0.0)
    # edge nodes: the below-ground plane copies node 0, so du/dz there is a/2
    edge = 1.1 * ((cb._CB_C_S ** 2 / cb._CB_PR_T) * delta2 * a / 2 + cb._CB_D_MOL)
    assert np.allclose(rk[..., 0], edge, rtol=1e-12) and np.allclose(rk[..., -1], edge)


def test_rhok_stable_layer_applies_the_buoyancy_factor():
    nx, ny, nz = 6, 6, 12
    dx, dy, dz = _grid(nx, ny, nz)
    a = 0.02
    z = np.arange(nz) * dz
    u = np.broadcast_to(a * z, (nx, ny, nz)).copy()
    zero = np.zeros_like(u)
    p = np.full_like(u, 1.0e5)                      # p = p00: theta = T exactly
    T = np.broadcast_to(290.0 + 0.003 * z, u.shape).copy()
    rho = np.ones_like(u)
    rk = cb.smagorinsky_lilly_rhok(u, zero, zero, T, zero, zero, p, rho, dx, dy, dz)
    k = 5
    n2 = cb._CB_G / T[0, 0, k] * 0.003
    ri = n2 / a ** 2
    fb = max(0.0, 1.0 - ri / cb._CB_PR_T) ** 0.25
    assert 0.0 < fb < 1.0
    want = (cb._CB_C_S ** 2 / cb._CB_PR_T) * (dx * dy * dz) ** (2 / 3) * fb ** 2 * a \
        + cb._CB_D_MOL
    assert np.isclose(rk[0, 0, k], want, rtol=1e-12)


def test_rhok_moist_layer_uses_the_runs_moist_virtual_temperature():
    nx, ny, nz = 6, 6, 12
    dx, dy, dz = _grid(nx, ny, nz)
    a = 0.02
    z = np.arange(nz) * dz
    u = np.broadcast_to(a * z, (nx, ny, nz)).copy()
    zero = np.zeros_like(u)
    p = np.full_like(u, 1.0e5)                      # p = p00: Exner factor 1
    T = np.full_like(u, 290.0)
    q_t = np.broadcast_to(0.008 + 2.0e-6 * z, u.shape).copy()   # moister aloft: stable
    rk = cb.smagorinsky_lilly_rhok(u, zero, zero, T, q_t, zero, p, np.ones_like(u),
                                   dx, dy, dz)
    k = 5
    r_d, r_v = 286.69, 461.5                        # the CloudBench run's values

    def theta_v(q):
        return (1.0 + (r_v / r_d - 1.0) * q) * 290.0

    dthv = (theta_v(q_t[0, 0, k + 1]) - theta_v(q_t[0, 0, k - 1])) / (2 * dz)
    ri = 9.81 / theta_v(q_t[0, 0, k]) * dthv / a ** 2
    assert ri > 0.0
    fb = max(0.0, 1.0 - ri / 0.3) ** 0.25
    want = (0.18 ** 2 / 0.3) * (dx * dy * dz) ** (2 / 3) * fb ** 2 * a + 0.1
    assert np.isclose(rk[0, 0, k], want, rtol=1e-12)


def test_sgs_flux_of_a_linear_profile_is_minus_k_gradient():
    nx, ny, nz = 4, 4, 10
    dz = 12.5
    z = np.arange(nz) * dz
    phi = np.broadcast_to(300.0 + 0.004 * z, (nx, ny, nz)).copy()
    rho = np.full_like(phi, 1.2)
    rhok = np.full_like(phi, 1.2 * 3.0)                # K = 3 m2/s
    f = cb.sgs_flux_nodes(rhok, rho, phi, dz)
    assert np.isnan(f[0]) and np.isnan(f[-1])
    assert np.allclose(f[1:-1], -3.0 * 0.004, rtol=1e-12)


def test_quick_face_flux_upwinds_and_is_exact_for_quadratics():
    nz = 9
    k = np.arange(nz, dtype=float)
    phi = (k ** 2)[None, None, :]                    # QUICK is exact for quadratics
    face = (k - 0.5) ** 2
    for m in (2.0, -3.0):
        f = cb.quick_face_flux(np.full_like(phi, m), phi)[0, 0]
        assert np.all(np.isnan(f[:2])) and np.isnan(f[-1])
        assert np.allclose(f[2:-1], m * face[2:-1], rtol=1e-12)


def _budget_case(sgs_factor):
    nx, ny, nz = 5, 5, 40
    dx, dy, dz = _grid(nx, ny, nz)
    rng = np.random.default_rng(3)
    phi = 300.0 + rng.normal(size=(nx, ny, nz))
    w = rng.normal(size=(nx, ny, nz))
    rho = np.full_like(phi, 1.1)
    rhok = 1.1 * (5.0 + rng.uniform(size=(nx, ny, nz)))
    q = cb.quick_face_flux(rho * w, phi)
    d_adv = np.full_like(phi, np.nan)
    d_adv[..., :-1] = (q[..., 1:] - q[..., :-1]) / dz
    fd = 0.5 * (rhok[..., 1:] + rhok[..., :-1]) * np.diff(phi, axis=-1) / dz
    d_sgs = np.full_like(phi, np.nan)
    d_sgs[..., 1:-1] = (fd[..., 1:] - fd[..., :-1]) / dz
    tend = -d_adv + sgs_factor * d_sgs
    z_phys = (np.arange(nz) + 0.5) * dz
    return cb.budget_check(rhok, rho, w, phi, np.nan_to_num(tend), dz, z_phys)


def test_budget_check_closes_when_the_tendency_is_advection_plus_rebuilt_sgs():
    good = _budget_case(1.0)
    assert good["ratio"] < 1e-12
    bad = _budget_case(2.0)                # the LES diffused twice as hard as rebuilt
    assert bad["ratio"] == pytest.approx(0.5, rel=1e-9)


def _sounding(saturated=False):
    z = np.array([0.0, 500.0, 1000.0, 3000.0, 6000.0])
    T = np.array([298.0, 294.0, 290.0, 278.0, 260.0])
    q = np.array([0.03 if saturated else 0.012, 0.010, 0.008, 0.003, 0.001])
    p = np.array([1.0e5, 9.45e4, 8.9e4, 7.0e4, 4.7e4])
    return {"z": z, "theta_li": T, "temperature": T, "T": T.copy(), "q_t": q,
            "u": np.array([-5.0, -6.0, -7.0, -4.0, 0.0]), "v": np.zeros(5), "p": p,
            "w": np.array([0.0, -0.002, -0.004, -0.004, 0.0]),
            "T_adv_src": np.full(5, -1.0e-5), "q_t_adv_src": np.full(5, -2.0e-8)}


def test_nudge_rate_is_swirl_lms_cosine_ramp_over_one_day():
    z = np.array([0.0, 2999.0, 3000.0, 3250.0, 3500.0, 5000.0])
    r = cb.nudge_rate(z) * 86400.0
    assert np.allclose(r, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-12)


def test_gcm_forcing_matches_swirl_lms_recipe_on_linear_profiles():
    z = np.array([0.0, 2000.0, 4000.0, 6000.0])
    w, gam, dq = -0.003, -0.0065, -1.0e-6
    s = {"z": z, "T": 300.0 + gam * z, "q_t": 0.01 + dq * z, "w": np.full(4, w),
         "T_adv_src": np.full(4, -1.0e-5), "q_t_adv_src": np.full(4, -2.0e-8)}
    zc = np.array([100.0, 3250.0, 5000.0])
    p_ref = np.array([9.9e4, 7.0e4, 5.5e4])
    f = cb.gcm_forcing(s, zc, p_ref)
    exner_inv = (p_ref / 1.0e5) ** (-cb._CB_R_D / cb._CB_CP_D)
    assert np.allclose(f["subsidence_w"], w)
    assert np.allclose(f["theta_adv"],
                       exner_inv * (-1.0e-5 + w * gam + w * cb._CB_G / cb._CB_CP_D),
                       rtol=1e-12)
    assert np.allclose(f["qv_adv"], -2.0e-8 + w * dq, rtol=1e-12, atol=0.0)
    assert np.allclose(f["nudge_rate"] * 86400.0, [0.0, 0.5, 1.0], atol=1e-12)


def test_gcm_forcing_takes_the_upwind_side_for_either_sign_of_w():
    z = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    T = 300.0 - 1.0e-6 * z**2
    w = np.array([0.0, 0.01, 0.01, -0.01, 0.0])
    s = {"z": z, "T": T, "q_t": 1.0e-3 * T, "w": w,
         "T_adv_src": np.zeros(5), "q_t_adv_src": np.zeros(5)}
    f = cb.gcm_forcing(s, z, np.full(5, 1.0e5))
    # donor below where w > 0 (1, 2 km), above where w < 0 (3 km)
    dTdz = np.array([(T[1] - T[0]), (T[2] - T[1]), (T[4] - T[3])]) / 1000.0
    want = w[1:4] * (dTdz + cb._CB_G / cb._CB_CP_D)
    assert np.allclose(f["theta_adv"][1:4], want, rtol=1e-12, atol=0.0)
    assert np.allclose(f["qv_adv"][1:4], 1.0e-3 * w[1:4] * dTdz, rtol=1e-12, atol=0.0)


def test_initial_profiles_interpolate_linearly_and_condense_with_our_curve():
    zp = np.array([250.0, 750.0, 2000.0])
    s = _sounding()
    out = cb.initial_profiles(s, zp)
    T = np.interp(zp, s["z"], s["T"])
    p = np.interp(zp, s["z"], s["p"])
    assert np.allclose(out["qc"], 0.0)
    assert np.allclose(out["theta"], T / np.asarray(exner_function(p)), rtol=1e-12)
    ws = _sounding(saturated=True)
    wet = cb.initial_profiles(ws, np.array([10.0]))
    t_w, p_w = np.interp(10.0, ws["z"], ws["T"]), np.interp(10.0, ws["z"], ws["p"])
    assert wet["qc"][0] > 0.0
    assert wet["theta"][0] < t_w / float(exner_function(p_w))   # latent term lowers theta_l
    s["temperature"] = s["T"] + 1.0
    with pytest.raises(ValueError, match="differ"):
        cb.initial_profiles(s, zp)


def _write_case(tmp_path, nx=6, ny=6, nz=16, nt=2, z_scale=1.0):
    import pandas as pd
    import xarray as xr
    dx, dy, dz = _grid(nx, ny, nz)
    rng = np.random.default_rng(0)
    z = np.arange(nz) * dz * z_scale
    shape = (nt, nx, ny, nz)
    zz = np.broadcast_to(z, shape)
    T = 295.0 - 0.006 * zz + 0.1 * rng.normal(size=shape)
    data = {
        "u": -5.0 + 0.001 * zz + 0.3 * rng.normal(size=shape),
        "v": 0.3 * rng.normal(size=shape), "w": 0.2 * rng.normal(size=shape),
        "theta_li": T + 2.0, "q_t": 0.01 * np.exp(-zz / 2000.0),
        "q_c": np.zeros(shape), "q_r": np.zeros(shape), "T": T,
        "rho": 1.15 * np.exp(-zz / 8000.0), "p_ref": 1.0e5 * np.exp(-zz / 8000.0),
        "rad_heat_src": -2.0e-5 * np.ones(shape),
        "theta_li_les_tendency": 1e-5 * rng.normal(size=shape),
        "q_t_les_tendency": 1e-8 * rng.normal(size=shape),
    }
    t = pd.to_timedelta([96 * 3600 + 1200 * i for i in range(nt)], unit="s")
    ds = xr.Dataset({k: (("t", "x", "y", "z"), v.astype("float32")) for k, v in data.items()},
                    coords={"t": t, "x": np.arange(nx) * dx, "y": np.arange(ny) * dy, "z": z})
    ds["cloud_fraction"] = (("t", "z"), np.full((nt, nz), 0.2, dtype="float32"))
    ds.attrs["_swirl_lm.pbtxt"] = "boundary_models {\n  z_0: 0.001\n}\nsurface_gustiness: 1.0\n"
    store = tmp_path / "data.zarr"
    ds.to_zarr(store, consolidated=True)
    fdir = tmp_path / "forcing" / "0" / "7" / "amip"
    fdir.mkdir(parents=True)
    s = _sounding()
    cols = ["z", "theta_li", "temperature", "q_t", "u", "v", "w", "T_adv_src",
            "q_t_adv_src", "T", "p"]
    lines = [",".join(cols)] + [",".join(f"{s[c][i]:.6e}" for c in cols)
                                for i in range(len(s["z"]))]
    (fdir / "sounding.csv").write_text("\n".join(lines) + "\n")
    (fdir / "parameters.json").write_text(json.dumps(
        {"experiment": "amip", "month": 7, "sst": 298.33, "latitude": 16.5}))
    return store, tmp_path / "forcing", dz, nz


def test_build_artifact_from_a_store(tmp_path):
    store, forcing, dz, nz = _write_case(tmp_path)
    art = cb.build_artifact(0, 7, "amip", forcing, store=str(store))
    assert art.case_name == "cloudbench_s000_m07_amip" and art.prescribe == "T_s"
    assert np.allclose(art.heights_m, (np.arange(1, nz - 1) + 0.5) * dz)
    assert np.allclose(art.times_s, [0.0, 96 * 3600.0, 96 * 3600.0 + 1200.0])
    assert np.allclose(art.T_s, 298.33)
    z_node = np.arange(1, nz - 1) * dz          # swirl-lm sounding coordinate k·dz
    s = _sounding()
    assert np.allclose(np.asarray(art.qt)[0], np.interp(z_node, s["z"], s["q_t"]))
    assert np.allclose(np.asarray(art.subsidence_w), np.interp(z_node, s["z"], s["w"]))
    assert np.allclose(np.asarray(art.nudge_rate), cb.nudge_rate(z_node))
    assert np.all(np.isfinite(art.theta_adv)) and np.all(np.isfinite(art.qv_adv))
    assert art.les_setup["forcing_recipe"]["tau_wind_s"] == 21600.0
    for k in ("wtheta_resolved", "wtheta_sgs", "wqt_resolved", "wqt_sgs"):
        a = np.asarray(getattr(art, k))
        assert np.all(a[0] == 0.0) and np.all(np.isfinite(a[1:])), k
    rad = np.asarray(art.rad_T_tendency)
    assert np.all(np.isnan(rad[0])) and np.allclose(rad[1:], -2.0e-5, rtol=1e-6)
    assert np.allclose(np.asarray(art.cloud_frac)[1:], 0.2, rtol=1e-6)
    checks = art.les_setup["budget_check"]
    assert len(checks) == 2 and set(checks[0]) == {"time_s", "theta_li", "q_t"}
    path = tmp_path / "a.npz"
    save_artifact(art, path)
    back = load_artifact(path)
    assert np.array_equal(np.asarray(back.rad_T_tendency), rad, equal_nan=True)
    assert np.array_equal(np.asarray(back.nudge_rate), np.asarray(art.nudge_rate))
    assert back.les_setup["height_offset_m"] == pytest.approx(0.5 * dz)
    from legoesm.atmosphere.les_suite.bridge import surface_roughness
    assert surface_roughness(back) == (0.001, 0.001)
    assert back.les_setup["surface_gustiness_m_s"] == 1.0
    from legoesm.atmosphere.les_suite.scm_runner import moist_scm_microphysics
    micro = moist_scm_microphysics(back)
    assert micro.scheme == "kessler" and micro.kessler.parameter_set == "cloudbench"


def test_held_out_sites_are_a_fixed_fifth_of_the_sites():
    sites = list(range(500))
    a = cb.held_out_sites(sites)
    assert len(a) == 100 and a == sorted(set(a)) and set(a) <= set(sites)
    assert a == cb.held_out_sites(list(reversed(sites)) + sites[:3])  # order/dupes
    assert a != cb.held_out_sites(sites, seed=cb.HELD_OUT_SEED + 1)


def test_run_surface_settings_refuses_a_missing_or_repeated_key():
    assert cb.run_surface_settings("z_0: 0.002\nsurface_gustiness: 0.5\n") == {
        "z_0": 0.002, "surface_gustiness": 0.5}
    with pytest.raises(ValueError, match="z_0 0 times"):
        cb.run_surface_settings("surface_gustiness: 0.5\n")
    with pytest.raises(ValueError, match="z_0 2 times"):
        cb.run_surface_settings("z_0: 0.002\n z_0: 0.003\nsurface_gustiness: 0.5\n")


def test_build_artifact_refuses_a_grid_that_is_not_cloudbench(tmp_path):
    store, forcing, _, _ = _write_case(tmp_path, z_scale=1.01)
    with pytest.raises(ValueError, match="grid differs"):
        cb.build_artifact(0, 7, "amip", forcing, store=str(store))


def test_artifact_refuses_a_misshapen_radiation_series(tmp_path):
    store, forcing, _, _ = _write_case(tmp_path)
    art = cb.build_artifact(0, 7, "amip", forcing, store=str(store), time_indices=[0])
    import dataclasses
    with pytest.raises(BridgeError, match="rad_T_tendency"):
        dataclasses.replace(art, rad_T_tendency=np.zeros((1, 3)))
