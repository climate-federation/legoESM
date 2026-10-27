"""CAM6 Mitchell ice/snow optics port (rrtmgp/optics/mitchell_ice_optics.py).

The reference below is an independent per-cell NumPy transcription of CESM2.1
CAM6 cloud_rad_props.F90 interpolate_ice_optics_sw (:472-520) and
interpolate_ice_optics_lw (:630-660) with interpolate_data.F90 lininterp_init's
extrap_method_bndry (:178-195): written from the Fortran, not from the JAX code.
"""
import hashlib
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)   # fp64 comparison against the Fortran transcription

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import mitchell_ice_optics as mio

RRTMG_SW_HIGH = np.array([3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000.,
                          22650., 29000., 38000., 50000., 2600.])


def _lininterp_bndry(yin, fin, y):
    """interpolate_data.F90 lininterp_init/lininterp, increasing yin, bndry extrap."""
    if y <= yin[0]:
        return fin[0]
    if y > yin[-1]:
        return fin[-1]
    for j in range(len(yin) - 1):
        if yin[j] < y <= yin[j + 1]:
            w = (yin[j + 1] - y) / (yin[j + 1] - yin[j])
            return w * fin[j] + (1.0 - w) * fin[j + 1]
    raise AssertionError("unreachable")


def _ref_sw(d_grid, ext, ssa, asm, path, dei):
    """Fortran loop: ext etc. are (n_d, nband) as CAM stores them."""
    nb = ext.shape[1]
    out = np.zeros((4, nb) + path.shape)
    for idx in np.ndindex(path.shape):
        if path[idx] < 1.e-80 or dei[idx] == 0.0:
            continue
        for b in range(nb):
            e = _lininterp_bndry(d_grid, ext[:, b], dei[idx])
            s = _lininterp_bndry(d_grid, ssa[:, b], dei[idx])
            a = _lininterp_bndry(d_grid, asm[:, b], dei[idx])
            tau = path[idx] * e
            out[0, b][idx] = tau
            out[1, b][idx] = tau * s
            out[2, b][idx] = tau * s * a
            out[3, b][idx] = tau * s * a * a
    return out


def _ref_lw(d_grid, absl, path, dei):
    nb = absl.shape[1]
    out = np.zeros((nb,) + path.shape)
    for idx in np.ndindex(path.shape):
        if path[idx] < 1.e-80 or dei[idx] == 0.0:
            continue
        for b in range(nb):
            out[b][idx] = min(path[idx] * _lininterp_bndry(d_grid, absl[:, b], dei[idx]), 50.0)
    return out


def _write_table(tmp_path, rng):
    import xarray as xr
    d = np.sort(rng.uniform(1.0, 750.0, 40))
    ds = xr.Dataset(
        {"sw_ext": (("sw_band", "d_eff"), rng.uniform(1.0, 200.0, (14, 40))),
         "sw_ssa": (("sw_band", "d_eff"), rng.uniform(0.5, 1.0, (14, 40))),
         "sw_asm": (("sw_band", "d_eff"), rng.uniform(0.7, 0.95, (14, 40))),
         # large enough that thick cells hit the 50 cap
         "lw_abs": (("lw_band", "d_eff"), rng.uniform(1.0, 400.0, (16, 40)))},
        coords={"d_eff": d},
        attrs={"SW_band_sequence_based_on_shortest_wavelength":
               ", ".join(f"{1e4 / w:.2f}" for w in RRTMG_SW_HIGH)})
    p = tmp_path / "iceoptics_synthetic.nc"
    ds.to_netcdf(p)
    return p, ds


@pytest.fixture
def table(tmp_path):
    rng = np.random.default_rng(7)
    p, ds = _write_table(tmp_path, rng)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return mio.load_mitchell_ice_optics(p, expected_sha256=sha), ds


def _cells(ds):
    d = ds["d_eff"].values
    # below the table, on a node, between nodes, above the table, D == 0, tiny path
    dei = np.array([[0.5, d[3], 0.5 * (d[10] + d[11]), 900.0],
                    [0.0, d[20] + 1.0, d[-1], 300.0]])
    path = np.array([[0.02, 0.3, 0.001, 0.05],
                     [0.05, 1.0e-81, 2.0, 0.4]])   # 2.0 kg/m2 x lw_abs hits the cap
    return path, dei


def test_sw_matches_fortran_transcription(table):
    tab, ds = table
    path, dei = _cells(ds)
    got = np.stack([np.asarray(x) for x in mio.mitchell_sw_optics(tab, path, dei)])
    ref = _ref_sw(ds["d_eff"].values, ds["sw_ext"].values.T, ds["sw_ssa"].values.T,
                  ds["sw_asm"].values.T, path, dei)
    ref = ref[:, np.r_[13, 0:13]]          # RRTMG order -> RRTMGP order (820 cm-1 band first)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0.0)
    assert np.all(got[:, :, 1, 0] == 0.0) and np.all(got[:, :, 1, 1] == 0.0)   # both guards


def test_lw_matches_fortran_transcription(table):
    tab, ds = table
    path, dei = _cells(ds)
    got = np.asarray(mio.mitchell_lw_abs_od(tab, path, dei))
    ref = _ref_lw(ds["d_eff"].values, ds["lw_abs"].values.T, path, dei)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0.0)
    assert got.max() == 50.0 and (ref == 50.0).any()   # the cap was exercised


def test_loader_refuses_wrong_checksum_and_band_order(tmp_path):
    rng = np.random.default_rng(3)
    p, ds = _write_table(tmp_path, rng)
    with pytest.raises(ValueError, match="sha256"):
        mio.load_mitchell_ice_optics(p, expected_sha256="0" * 64)
    bad = ds.copy()
    bad.attrs["SW_band_sequence_based_on_shortest_wavelength"] = ", ".join(
        f"{1e4 / w:.2f}" for w in np.roll(RRTMG_SW_HIGH, 1))
    q = tmp_path / "bad.nc"
    bad.to_netcdf(q)
    with pytest.raises(ValueError, match="band sequence"):
        mio.load_mitchell_ice_optics(q, expected_sha256=hashlib.sha256(open(q, "rb").read()).hexdigest())
    nanv = ds["lw_abs"].values.copy()
    nanv[2, 5] = np.inf
    r = tmp_path / "nan.nc"
    ds.assign(lw_abs=(ds.lw_abs.dims, nanv)).to_netcdf(r)
    with pytest.raises(ValueError, match="non-finite"):
        mio.load_mitchell_ice_optics(r, expected_sha256=hashlib.sha256(open(r, "rb").read()).hexdigest())


def test_gradient_finite_and_matches_fd(table):
    import jax.numpy as jnp
    tab, ds = table
    d0 = 0.5 * (ds["d_eff"].values[10] + ds["d_eff"].values[11])

    def f(path, d):
        tau, tau_w, _, tau_w_f = mio.mitchell_sw_optics(tab, path, d)
        return jnp.sum(tau + tau_w_f) + jnp.sum(mio.mitchell_lw_abs_od(tab, path, d))

    gp, gd = jax.grad(f, argnums=(0, 1))(0.01, d0)
    eps = 1e-6
    fd_p = (f(0.01 + eps, d0) - f(0.01 - eps, d0)) / (2 * eps)
    fd_d = (f(0.01, d0 + eps) - f(0.01, d0 - eps)) / (2 * eps)
    np.testing.assert_allclose([gp, gd], [fd_p, fd_d], rtol=1e-6)
    # zero path (guarded) and D == 0: finite gradients
    for args in ((0.0, d0), (0.01, 0.0)):
        assert np.all(np.isfinite(jax.grad(f, argnums=(0, 1))(*args)))


@pytest.mark.skipif(not os.environ.get("LEGOESM_MITCHELL_ICEOPTICS"),
                    reason="real CAM6 iceoptics_c080917.nc not configured")
def test_real_cam6_table_loads_and_rotates():
    import xarray as xr
    p = os.environ["LEGOESM_MITCHELL_ICEOPTICS"]
    tab = mio.load_mitchell_ice_optics(p)
    with xr.open_dataset(p) as ds:
        np.testing.assert_array_equal(np.asarray(tab.sw_ext[0]), ds["sw_ext"].values[13])
        np.testing.assert_array_equal(np.asarray(tab.sw_ext[1]), ds["sw_ext"].values[0])
        np.testing.assert_array_equal(np.asarray(tab.lw_abs), ds["lw_abs"].values)


def test_gradient_bracket_and_fp32_zero_path(table):
    """AD follows CAM's bracket x[j] < y <= x[j+1]: at an interior knot the
    slope is the LEFT interval's; at the first knot (and below) zero.  In fp32 a
    zero path still has zero optics and zero path gradient."""
    import jax.numpy as jnp
    tab, ds = table
    d = ds["d_eff"].values
    ext = ds["sw_ext"].values[0]                      # RRTMG band 1 = RRTMGP band 2
    g = jax.grad(lambda y: mio.mitchell_sw_optics(tab, 1.0, y)[0][1])
    k = 10
    left = (ext[k] - ext[k - 1]) / (d[k] - d[k - 1])
    np.testing.assert_allclose(g(float(d[k])), left, rtol=1e-12)
    assert g(float(d[0])) == 0.0 and g(float(d[0]) - 0.1) == 0.0
    t32 = mio.MitchellIceOptics(*(jnp.asarray(a, dtype=jnp.float32) for a in tab))
    gp = jax.grad(lambda p: jnp.sum(mio.mitchell_sw_optics(t32, p, jnp.float32(50.0))[0])
                  + jnp.sum(mio.mitchell_lw_abs_od(t32, p, jnp.float32(50.0))))(jnp.float32(0.0))
    assert gp == 0.0


def test_off_cells_never_read_the_table(table):
    """CAM skips an off cell entirely: zero path with a NaN diameter gives zero
    optics in every SW quantity and in LW; the smallest positive fp32 path XLA
    represents is ON (it exceeds CAM's 1e-80; XLA CPU flushes subnormals to zero,
    so a subnormal path is a zero path here)."""
    import jax.numpy as jnp
    tab, _ = table
    sw = mio.mitchell_sw_optics(tab, np.array([0.0]), np.array([np.nan]))
    assert all(np.all(np.asarray(x) == 0.0) for x in sw)
    assert np.all(np.asarray(mio.mitchell_lw_abs_od(tab, np.array([0.0]), np.array([np.nan]))) == 0.0)
    t32 = mio.MitchellIceOptics(*(jnp.asarray(a, dtype=jnp.float32) for a in tab))
    small = jnp.float32(np.finfo(np.float32).tiny)  # smallest normal fp32
    assert float(mio.mitchell_sw_optics(t32, small, jnp.float32(50.0))[0][0]) > 0.0


def test_off_cell_gradient_is_finite_with_nan_diameter(table):
    import jax.numpy as jnp
    tab, _ = table

    def f(p, d):
        return (sum(jnp.sum(x) for x in mio.mitchell_sw_optics(tab, p, d))
                + jnp.sum(mio.mitchell_lw_abs_od(tab, p, d)))
    gp, gd = jax.grad(f, argnums=(0, 1))(0.0, jnp.nan)
    assert gp == 0.0 and gd == 0.0
