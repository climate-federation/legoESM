"""Unit tests for the ML ozone parameterization (Ma et al. ridge regression).

Synthesizes a tiny set of NetCDF weight files matching the upstream layout
to exercise the loader, the JAX forward pass, AD, dispatch through
``_compute_ozone_vmr``, and the unknown-literal guard.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation import (
    MLOzoneCoefficients,
    OzoneProfileConfig,
    load_ml_ozone_coefficients,
    predict_ozone_ml,
)
from legoesm.atmosphere.physics.radiation.integration import _compute_ozone_vmr


# ---------------------------------------------------------------------------
# Helpers: build a tiny synthetic weight directory.
# ---------------------------------------------------------------------------

def _make_synthetic_weights(tmpdir: Path, n_lev=6, n_lat=4, n_lon=8, seed=0):
    """Write coefs/Scaler_x/Scaler_y/plev NetCDFs to ``tmpdir``."""
    netCDF4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(seed)

    # Ridge mapping: identity-ish to make round-trip predictable.
    # y_norm = x_norm @ I  ==>  with x = x_mean, y = y_mean (zero residual).
    coefs = np.zeros((n_lev, n_lev, n_lat, n_lon), dtype=np.float64)
    for c in range(n_lev):
        coefs[c, c, :, :] = 1.0
    # Add small noise to other entries so the matmul is not trivial.
    coefs += 1e-3 * rng.standard_normal(coefs.shape)
    # Zero the noise on the diagonal to keep predictability for the identity test.
    for c in range(n_lev):
        coefs[c, c, :, :] = 1.0

    # Temperature climatology: 280 - 5 * lev (warmer at surface).
    z = np.arange(n_lev, dtype=np.float64)
    x_mean = np.broadcast_to(
        (280.0 - 5.0 * z)[:, None, None], (n_lev, n_lat, n_lon)
    ).copy()
    x_scale = np.full((n_lev, n_lat, n_lon), 5.0, dtype=np.float64)

    # Ozone climatology: Gaussian in level, plus weak lat dependence.
    lev_peak = n_lev // 2
    y_mean = np.broadcast_to(
        (5e-6 * np.exp(-0.5 * ((z - lev_peak) / 1.5) ** 2))[:, None, None],
        (n_lev, n_lat, n_lon),
    ).copy()
    y_scale = np.full((n_lev, n_lat, n_lon), 1e-6, dtype=np.float64)

    p_uk = np.linspace(100.0, 100_000.0, n_lev, dtype=np.float64)  # Pa, ascending

    # Write NetCDFs.
    def _write(path, vars_):
        with netCDF4.Dataset(str(path), "w", format="NETCDF4_CLASSIC") as ds:
            for name, arr in vars_.items():
                if name in ("x_mean", "x_scale", "y_mean", "y_scale"):
                    dims = ("z", "lat", "lon")
                    sizes = (n_lev, n_lat, n_lon)
                elif name == "coefs":
                    dims = ("c", "z", "lat", "lon")
                    sizes = (n_lev, n_lev, n_lat, n_lon)
                for dname, dsize in zip(dims, sizes):
                    if dname not in ds.dimensions:
                        ds.createDimension(dname, dsize)
                v = ds.createVariable(name, np.float64, dims)
                v[:] = arr

    _write(tmpdir / "coefs_synth.nc", {"coefs": coefs})
    _write(tmpdir / "Scaler_x_synth.nc", {"x_mean": x_mean, "x_scale": x_scale})
    _write(tmpdir / "Scaler_y_synth.nc", {"y_mean": y_mean, "y_scale": y_scale})
    np.save(tmpdir / "plev.npy", p_uk)
    return dict(coefs=coefs, x_mean=x_mean, x_scale=x_scale,
                y_mean=y_mean, y_scale=y_scale, p_uk=p_uk)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loader_round_trips_arrays(tmp_path):
    truth = _make_synthetic_weights(tmp_path)
    coefs = load_ml_ozone_coefficients(tmp_path)
    assert isinstance(coefs, MLOzoneCoefficients)
    assert coefs.coefs.shape == truth["coefs"].shape
    np.testing.assert_allclose(np.asarray(coefs.x_mean), truth["x_mean"])
    np.testing.assert_allclose(np.asarray(coefs.p_uk), truth["p_uk"])


def test_identity_T_equals_mean_returns_y_mean(tmp_path):
    truth = _make_synthetic_weights(tmp_path)
    coefs = load_ml_ozone_coefficients(tmp_path)
    n_lev = truth["coefs"].shape[0]

    # Build a single model column on the UKESM levels with T == x_mean exactly.
    p_full = coefs.p_uk[None, :]                  # (1, n_lev)
    T = jnp.asarray(truth["x_mean"][:, 0, 0])[None, :]  # use (j=0, i=0) clim
    # Place the column at lat_uk[0], lon_uk[0] so bilinear weights collapse.
    lat = jnp.asarray([np.deg2rad(float(coefs.lat_uk[0]))])
    lon = jnp.asarray([np.deg2rad(float(coefs.lon_uk[0]))])

    o3 = predict_ozone_ml(T, lat, lon, p_full, coefs, mmr_to_vmr=False)
    expected = truth["y_mean"][:, 0, 0]
    o3_np = np.asarray(o3[0])
    # Allow a small slack due to the off-diagonal coef noise (zeroed but
    # still finite-precision) and vertical interp.
    assert np.allclose(o3_np, expected, rtol=1e-6, atol=1e-9), (
        f"o3={o3_np}, expected={expected}"
    )


def test_predicted_o3_is_finite_and_nonnegative(tmp_path):
    _make_synthetic_weights(tmp_path)
    coefs = load_ml_ozone_coefficients(tmp_path)
    n_lev = coefs.p_uk.shape[0]
    ncol = 5
    rng = np.random.default_rng(42)
    T = jnp.asarray(280.0 + 10.0 * rng.standard_normal((ncol, n_lev)))
    p_full = jnp.broadcast_to(coefs.p_uk, (ncol, n_lev))
    lat = jnp.asarray(np.deg2rad(rng.uniform(-89, 89, ncol)))
    lon = jnp.asarray(np.deg2rad(rng.uniform(0, 360, ncol)))

    o3 = predict_ozone_ml(T, lat, lon, p_full, coefs, mmr_to_vmr=False)
    assert o3.shape == (ncol, n_lev)
    assert jnp.all(jnp.isfinite(o3))
    assert jnp.all(o3 >= 0.0)


def test_predict_is_differentiable_wrt_T(tmp_path):
    _make_synthetic_weights(tmp_path)
    coefs = load_ml_ozone_coefficients(tmp_path)
    n_lev = coefs.p_uk.shape[0]
    ncol = 3
    T = jnp.asarray(np.full((ncol, n_lev), 270.0))
    p_full = jnp.broadcast_to(coefs.p_uk, (ncol, n_lev))
    lat = jnp.asarray(np.deg2rad(np.array([10.0, 30.0, -50.0])))
    lon = jnp.asarray(np.deg2rad(np.array([20.0, 200.0, 359.0])))

    def loss(T_):
        o3 = predict_ozone_ml(T_, lat, lon, p_full, coefs, mmr_to_vmr=False)
        return jnp.sum(o3 ** 2)

    g = jax.grad(loss)(T)
    assert g.shape == T.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


def test_mmr_to_vmr_conversion(tmp_path):
    _make_synthetic_weights(tmp_path)
    coefs = load_ml_ozone_coefficients(tmp_path)
    n_lev = coefs.p_uk.shape[0]
    T = jnp.full((1, n_lev), 270.0)
    p_full = coefs.p_uk[None, :]
    lat = jnp.zeros(1); lon = jnp.zeros(1)
    o3_mmr = predict_ozone_ml(T, lat, lon, p_full, coefs, mmr_to_vmr=False)
    o3_vmr = predict_ozone_ml(T, lat, lon, p_full, coefs, mmr_to_vmr=True)
    ratio = constants.M_dry / constants.M_o3
    # Where above the floor, ratio should hold.
    above_floor = (o3_mmr > 1e-9)
    np.testing.assert_allclose(
        np.asarray(o3_vmr)[np.asarray(above_floor)],
        ratio * np.asarray(o3_mmr)[np.asarray(above_floor)],
        rtol=1e-6,
    )


def test_compute_ozone_vmr_ml_branch(tmp_path):
    _make_synthetic_weights(tmp_path)
    coefs = load_ml_ozone_coefficients(tmp_path)
    n_lev = coefs.p_uk.shape[0]
    ncol = 2
    T = jnp.full((ncol, n_lev), 270.0)
    p_full = jnp.broadcast_to(coefs.p_uk, (ncol, n_lev))
    lat = jnp.zeros(ncol)
    lon = jnp.zeros(ncol)
    cfg = OzoneProfileConfig(source="ml", ml_weights_path=str(tmp_path),
                              ml_mmr_to_vmr=True)
    o3 = _compute_ozone_vmr(p_full, lat, cfg, T=T, lon=lon, ml_ozone_coefs=coefs)
    assert o3.shape == (ncol, n_lev)
    assert jnp.all(jnp.isfinite(o3))


def test_compute_ozone_vmr_rejects_unknown_source():
    p_full = jnp.zeros((1, 4))
    lat = jnp.zeros(1)
    cfg = OzoneProfileConfig(source="bogus")
    with pytest.raises(ValueError, match="Unknown OzoneProfileConfig.source"):
        _compute_ozone_vmr(p_full, lat, cfg)


def test_compute_ozone_vmr_ml_missing_inputs_raises(tmp_path):
    cfg = OzoneProfileConfig(source="ml", ml_weights_path=str(tmp_path))
    p_full = jnp.zeros((1, 4))
    lat = jnp.zeros(1)
    with pytest.raises(ValueError, match="requires T, lon, and ml_ozone_coefs"):
        _compute_ozone_vmr(p_full, lat, cfg)
