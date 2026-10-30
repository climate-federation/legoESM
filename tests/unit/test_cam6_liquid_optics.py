"""CAM6 gammadist liquid optics port (rrtmgp/optics/cam6_liquid_optics.py) and
the shared CAM lininterp (rrtmgp/optics/cam_lininterp.py).

Reference = an independent per-cell NumPy transcription of CESM2.1 CAM6
utils/interpolate_data.F90 lininterp_init (extrap_method_bndry :178-212, interior
search :263-286) and lininterp2d1d (:400-438), and rrtmg/cloud_rad_props.F90
get_mu_lambda_weights (:745-769), gam_liquid_sw/lw (:668-741) with the lamc > 0
caller skip (:546, :580) and the 1/0.9970449e3 table conversion (:178-179).
"""
import hashlib
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)   # fp64 comparison against the Fortran transcription

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cam6_liquid_optics as clo  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cam_lininterp  # noqa: E402

SW_LOW = np.array([2600., 3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000.,
                   22650., 29000., 38000., 820.])
SW_HIGH = np.array([3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000., 22650.,
                    29000., 38000., 50000., 2600.])
LW_LOW = np.array([10., 350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800.,
                   2080., 2250., 2390., 2600.])
LW_HIGH = np.array([350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800., 2080.,
                    2250., 2390., 2600., 3250.])


def _f_init(yin, y):
    """lininterp_init, one output point, extrap_method_bndry -> (jjm, jjp, wgts, wgtn), 0-based."""
    n = len(yin)
    increasing = not (yin[n - 1] < yin[0])
    jjm = jjp = None
    if increasing:
        if y <= yin[0]:
            jjm, jjp, ws, wn = 0, 0, 1.0, 0.0
        elif y > yin[n - 1]:
            jjm, jjp, ws, wn = n - 1, n - 1, 1.0, 0.0
    else:
        if y > yin[0]:
            jjm, jjp, ws, wn = 0, 0, 1.0, 0.0
        elif y <= yin[n - 1]:
            jjm, jjp, ws, wn = n - 1, n - 1, 1.0, 0.0
    for jj in range(n - 1):
        hit = (y > yin[jj] and y <= yin[jj + 1]) if increasing else (y <= yin[jj] and y > yin[jj + 1])
        if hit:
            jjm, jjp = jj, jj + 1
            ws = (yin[jj + 1] - y) / (yin[jj + 1] - yin[jj])
            wn = (y - yin[jj]) / (yin[jj + 1] - yin[jj])
            break
    assert jjm is not None
    return jjm, jjp, ws, wn


def _ref(g_mu, g_lambda, tabs, path, lamc, mu):
    """gam_liquid_sw/lw per cell; tabs = list of (nmu, nlambda, nband) arrays."""
    out = [np.zeros((t.shape[2],) + path.shape) for t in tabs]
    for idx in np.ndindex(path.shape):
        if not lamc[idx] > 0.0 or path[idx] < 1.e-80:
            continue
        im, ip, ww, we = _f_init(g_mu, mu[idx])
        lam_i = np.array([g_lambda[im, l] * ww + g_lambda[ip, l] * we
                          for l in range(g_lambda.shape[1])])
        jm, jp, ws, wn = _f_init(lam_i, lamc[idx])
        for k, t in enumerate(tabs):
            for b in range(t.shape[2]):
                a = t[:, :, b]
                out[k][b][idx] = (a[im, jm] * ww * ws + a[ip, jm] * we * ws
                                  + a[im, jp] * ww * wn + a[ip, jp] * we * wn)
    return out


def _write_table(tmp_path, rng):
    import xarray as xr
    nmu, nlam = 6, 9
    mu = np.sort(rng.uniform(2.0, 15.0, nmu))
    lam = np.sort(rng.uniform(6.0e4, 8.0e6, (nmu, nlam)), axis=1)[:, ::-1]   # decreasing
    sh = (nlam, nmu)
    ds = xr.Dataset(
        {"lambda": (("lambda_scale", "mu"), lam.T),
         "k_ext_sw": (("sw_band", "lambda_scale", "mu"), rng.uniform(1e4, 2e5, (14,) + sh)),
         "ssa_sw": (("sw_band", "lambda_scale", "mu"), rng.uniform(0.5, 1.0, (14,) + sh)),
         "asm_sw": (("sw_band", "lambda_scale", "mu"), rng.uniform(0.7, 0.9, (14,) + sh)),
         "k_abs_lw": (("lw_band", "lambda_scale", "mu"), rng.uniform(1e4, 2e5, (16,) + sh)),
         "wvl_sw_min": (("sw_band",), 1e-2 / SW_HIGH), "wvl_sw_max": (("sw_band",), 1e-2 / SW_LOW),
         "wvl_min_lw": (("lw_band",), 1e-2 / LW_HIGH), "wvl_max_lw": (("lw_band",), 1e-2 / LW_LOW)},
        coords={"mu": mu})
    p = tmp_path / "liq_synthetic.nc"
    ds.to_netcdf(p)
    return p, ds


@pytest.fixture
def table(tmp_path):
    p, ds = _write_table(tmp_path, np.random.default_rng(11))
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return clo.load_cam6_liquid_optics(p, expected_sha256=sha), ds


def _fortran_layout(ds, name):
    return np.transpose(ds[name].values, (2, 1, 0)) / (
        0.9970449e3 if name in ("k_ext_sw", "k_abs_lw") else 1.0)   # (nmu, nlambda, nband)


def _cells(ds):
    mu = ds["mu"].values
    lam = ds["lambda"].values.T                    # (nmu, nlambda)
    mu_c = np.array([[1.0, mu[2], 0.5 * (mu[3] + mu[4]), 20.0, mu[1]],
                     [mu[0], 0.5 * (mu[0] + mu[1]), mu[-1], 7.0, 9.0]])
    lam_c = np.array([[2.0e5, lam[2, 4], 9.0e6, 3.0e4, 0.0],
                      [lam[0, 0], 0.5 * (lam[0, 3] + lam[1, 3]), 1.0e6, -5.0, 4.0e5]])
    path = np.array([[0.02, 0.3, 0.001, 0.05, 0.1],
                     [0.05, 1.0e-81, 2.0, 0.4, 0.07]])
    return path, lam_c, mu_c


def test_sw_lw_match_fortran_transcription(table):
    tab, ds = table
    path, lamc, mu = _cells(ds)
    tabs = [_fortran_layout(ds, n) for n in ("k_ext_sw", "ssa_sw", "asm_sw", "k_abs_lw")]
    ext, ssa, asm, absl = _ref(ds["mu"].values, ds["lambda"].values.T, tabs, path, lamc, mu)
    tau = path * ext
    ref_sw = np.stack([tau, tau * ssa, tau * ssa * asm, tau * ssa * asm * asm])
    ref_sw = np.where((lamc > 0) & (path >= 1e-80), ref_sw, 0.0)[:, np.r_[13, 0:13]]
    got_sw = np.stack([np.asarray(x) for x in clo.cam6_liquid_sw_optics(tab, path, lamc, mu)])
    np.testing.assert_allclose(got_sw, ref_sw, rtol=1e-13, atol=0.0)
    ref_lw = np.where((lamc > 0) & (path >= 1e-80), path * absl, 0.0)
    np.testing.assert_allclose(np.asarray(clo.cam6_liquid_lw_abs_od(tab, path, lamc, mu)),
                               ref_lw, rtol=1e-13, atol=0.0)
    assert np.all(got_sw[:, :, 0, 4] == 0.0) and np.all(got_sw[:, :, 1, 3] == 0.0)  # lamc <= 0
    assert np.all(got_sw[:, :, 1, 1] == 0.0)                                       # tiny path
    assert np.asarray(clo.cam6_liquid_lw_abs_od(tab, path, lamc, mu)).max() > 50.0  # no LW cap


def test_lininterp_weights_both_directions():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0.0, 10.0, 7))
    for grid, inc in ((x, True), (x[::-1].copy(), False)):
        for y in (grid[0], grid[3], grid[-1], -1.0, 11.0, 0.5 * (grid[2] + grid[3])):
            want = _f_init(grid, y)
            got = tuple(np.asarray(v) for v in cam_lininterp.bndry_weights(grid, np.float64(y), inc))
            assert (int(got[0]), int(got[1])) == want[:2], (inc, y)
            np.testing.assert_allclose(got[2:], want[2:], rtol=1e-15, atol=0.0)


def test_loader_refuses_bad_files(tmp_path):
    p, ds = _write_table(tmp_path, np.random.default_rng(2))
    with pytest.raises(ValueError, match="sha256"):
        clo.load_cam6_liquid_optics(p, expected_sha256="0" * 64)
    nan_ext = ds["k_ext_sw"].values.copy()
    nan_ext[3, 2, 1] = np.nan
    for mut, msg in ((lambda d: d.assign(wvl_sw_min=d.wvl_sw_min[::-1]), "band limits"),
                     (lambda d: d.assign({"lambda": d["lambda"][::-1]}), "decreasing"),
                     (lambda d: d.isel(mu=[0]), ">= 2 points"),
                     (lambda d: d.assign(mu=d.mu[::-1].values), "increasing"),
                     (lambda d: d.assign(k_ext_sw=(d.k_ext_sw.dims, nan_ext)), "non-finite")):
        q = tmp_path / f"bad_{abs(hash(msg))}.nc"
        mut(ds).to_netcdf(q)
        with pytest.raises(ValueError, match=msg):
            clo.load_cam6_liquid_optics(q, expected_sha256=hashlib.sha256(open(q, "rb").read()).hexdigest())


def test_gradients(table):
    import jax.numpy as jnp
    tab, ds = table
    mu0 = 0.5 * (ds["mu"].values[2] + ds["mu"].values[3])
    lam0 = 3.0e5

    def f(p, l, m):
        return (sum(jnp.sum(x) for x in clo.cam6_liquid_sw_optics(tab, p, l, m))
                + jnp.sum(clo.cam6_liquid_lw_abs_od(tab, p, l, m)))
    g = jax.grad(f, argnums=(0, 1, 2))(0.01, lam0, mu0)
    for k, (h, args) in enumerate(((1e-7, lambda e: (0.01 + e, lam0, mu0)),
                                   (1.0, lambda e: (0.01, lam0 + e, mu0)),
                                   (1e-6, lambda e: (0.01, lam0, mu0 + e)))):
        fd = (f(*args(h)) - f(*args(-h))) / (2 * h)
        np.testing.assert_allclose(g[k], fd, rtol=1e-6)
    # off cells (zero path, lamc <= 0, NaN size): exact zero gradients
    for args in ((0.0, jnp.nan, jnp.nan), (0.01, 0.0, mu0), (0.01, -1.0, jnp.nan)):
        assert all(float(v) == 0.0 for v in jax.grad(f, argnums=(0, 1, 2))(*args))


@pytest.mark.skipif(not os.environ.get("LEGOESM_CAM6_LIQOPTICS"),
                    reason="real CAM6 F_nwvl200_mu20_lam50_res64_t298_c080428.nc not configured")
def test_real_cam6_table_loads_and_maps():
    import xarray as xr
    p = os.environ["LEGOESM_CAM6_LIQOPTICS"]
    tab = clo.load_cam6_liquid_optics(p)
    with xr.open_dataset(p) as ds:
        raw = ds["k_ext_sw"].transpose("sw_band", "mu", "lambda_scale").values / 0.9970449e3
        np.testing.assert_allclose(np.asarray(tab.sw_ext[0]), raw[13], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(tab.sw_ext[1]), raw[0], rtol=1e-15)
        assert tab.g_lambda.shape == (ds.sizes["mu"], ds.sizes["lambda_scale"])
