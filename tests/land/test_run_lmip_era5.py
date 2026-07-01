"""Tests for the bare-minimum ERA5-forced forward LMIP driver + the
``forward_ml(return_diag=True)`` extension it relies on.

The forward-shape test runs the REAL multilayer physics on a tiny synthetic
`data` dict with a small spin-up (globals patched down); the writer test mocks
the forward so it exercises the driver's assembly + NetCDF output cheaply.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import scripts.run.train_multilayer_land_era5 as TML
import scripts.run.run_lmip_era5 as D
from legoesm.core.coupling_fields import AtmToSurface

pytest.importorskip("xarray")

_NCOL, _NH = 3, 2


def _atm(nh, ncol, **over):
    base = dict(
        sw_down=200.0, lw_down=320.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=285.0, q_lowest=5e-3, u_lowest=3.0, v_lowest=0.0,
        p_lowest=0.99e5, p_surface=1.0e5, rho_lowest=1.22,
        cos_zenith=0.5, co2_ppmv=412.0, has_radiation=1.0, has_precipitation=1.0)
    base.update(over)
    return AtmToSurface(**{k: jnp.full((nh, ncol), v, dtype=jnp.float64)
                           for k, v in base.items()})


def _synthetic_data(ncol=_NCOL, nh=_NH):
    pft = np.zeros((ncol, 17)); pft[:, 4] = 1.0                 # broadleaf-evergreen
    const = lambda v: jnp.full(ncol, v, dtype=jnp.float64)
    data = dict(
        forc=[_atm(nh, ncol) for _ in range(12)],
        lat=jnp.deg2rad(jnp.linspace(-40.0, 40.0, ncol)),
        lat_deg=jnp.linspace(-40.0, 40.0, ncol), lon_deg=jnp.linspace(0.0, 300.0, ncol),
        pft=jnp.asarray(pft), fg=const(0.0), fc=const(0.30),
        pct_sand=const(40.0), pct_clay=const(20.0), t0=const(285.0),
        skt=jnp.full((12, ncol), 285.0), w=jnp.cos(jnp.deg2rad(jnp.linspace(-40, 40, ncol))),
    )
    for k, v in dict(theta_r=0.05, theta_sat=0.45, alpha_vg=2.0, n_vg=1.5, K_sat=1e-6).items():
        data["vg_" + k] = const(v)
    return data


def test_forward_ml_return_diag_shapes(monkeypatch):
    # small spin-up so the real physics run is quick
    monkeypatch.setattr(TML, "_NH", _NH)
    monkeypatch.setattr(TML, "_SPM", _NH)          # 1 representative day
    monkeypatch.setattr(TML, "_DT", 86400.0 / _NH)
    monkeypatch.setattr(TML, "_EQ_STEPS", 2)
    data = _synthetic_data()
    cp = TML.constrain_ext(TML.init_ext_params())

    out = TML.forward_ml(cp, data, return_diag=True)
    assert len(out) == 5
    T, A, SH, LH, st = out
    for arr in (T, A, SH, LH):
        assert np.asarray(arr).shape == (12, _NCOL)
        assert np.all(np.isfinite(np.asarray(arr)))
    assert np.asarray(st.T_soil).shape[0] == _NCOL          # (ncol, n_layers)
    assert np.asarray(st.theta_soil).shape[0] == _NCOL

    # backward-compat: default call still returns just (T, A)
    T2, A2 = TML.forward_ml(cp, data)
    assert np.allclose(np.asarray(T2), np.asarray(T))


def test_driver_writes_netcdf(monkeypatch, tmp_path):
    data = _synthetic_data()

    class _St:
        T_soil = jnp.full((_NCOL, 8), 285.0)
        theta_soil = jnp.full((_NCOL, 8), 0.3)

    monkeypatch.setattr(TML, "load_training_data", lambda *a, **k: data)
    monkeypatch.setattr(TML, "_NH", _NH); monkeypatch.setattr(TML, "_SPM", _NH)
    z = jnp.full((12, _NCOL), 285.0)
    monkeypatch.setattr(TML, "forward_ml",
                        lambda cp, d, return_diag=False: (z, z * 0 + 0.2, z * 0 + 30.0,
                                                          z * 0 + 40.0, _St()))
    npz = tmp_path / "era5.npz"; npz.write_bytes(b"stub")     # existence check only
    out = tmp_path / "lmip.nc"
    args = D.build_parser().parse_args(
        ["--diurnal-npz", str(npz), "--n-sub", "3", "--out", str(out)])
    rc = D.run(args)
    assert rc == 0
    import xarray as xr
    ds = xr.open_dataset(out)
    assert ds["T_sfc"].shape == (12, _NCOL)
    assert "shflx" in ds and "lhflx" in ds and "theta_soil_final" in ds
    assert ds.attrs["forcing"].startswith("ERA5")


def test_driver_missing_npz_exits_cleanly(tmp_path):
    args = D.build_parser().parse_args(
        ["--diurnal-npz", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "o.nc")])
    assert D.run(args) == 2
