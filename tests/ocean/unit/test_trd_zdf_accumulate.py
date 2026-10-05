"""``--trd-accumulate``: the implicit vertical-diffusion T tendency (NEMO
``ttrd_zdf``) handed from the tripole/lat-lon solve to a host accumulator."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_N_LAT, _N_LON, _NLEV, _H, _DT = 8, 16, 6, 600.0, 1800.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def _setup():
    from legoesm.core.field import Field
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig, SurfaceTracerForcing
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, TKEConfig
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z = create_ocean_z_star(n_levels=_NLEV, H_max=_H)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    st = rest_state_latlon_cgrid_ocean(grid, z, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.full((_N_LAT, _N_LON), _H))
    T = 25.0 - 3.0 * jnp.arange(_NLEV)[None, None, :] * jnp.ones((_N_LAT, _N_LON, 1))
    st = st._replace(T=st.T.replace(data=T), S=st.S.replace(data=jnp.full_like(T, 35.0)))
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=TKEConfig(prognostic=True)),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, implicit_vertical_mixing=True, enable_runtime_checks=False,
        barotropic_solver="rigid_lid", outer_integrator="forward_euler",
        K_v=1.0e-2, physics=physics)
    m = LatLonCGridOceanModel(grid, z, cfg)
    m._ensure_rigid_lid_data(st)
    tke0 = (jnp.where(lm[..., None] > 0.5, 1.0e-3, 0.0) * jnp.ones((1, 1, _NLEV - 1)))
    rate = np.zeros((_N_LAT, _N_LON, _NLEV)); rate[:, :, 0] = 1.0e-4 * lm
    src = SurfaceTracerForcing(
        dT_dt=Field(data=jnp.asarray(rate), name="dT", dims=("lat", "lon", "level"), units="degC/s"),
        dS_dt=Field(data=jnp.zeros((_N_LAT, _N_LON, _NLEV)), name="dS",
                    dims=("lat", "lon", "level"), units="PSU/s"))
    dz = np.asarray(compute_layer_thickness(st.eta.data, st.H_bathy.data, z))
    return m, st, tke0, src, rate, lm, dz


def test_emitted_tendency_is_the_solve_alone_and_conserves_column_heat():
    m, st, tke0, src, rate, lm, dz = _setup()
    got = []
    out, _ = m._apply_implicit_vertical_mixing(
        st, _DT, None, tke_old=tke0, return_tke=True,
        surface_tracer_forcing=src, trd_callback=lambda x, k: got.append(np.asarray(x)))
    assert len(got) == 1
    trd = got[0]
    T0, T1 = np.asarray(st.T.data), np.asarray(out.T.data)
    wet = lm > 0.5
    # The surface source is EXCLUDED: trd = (T1 - (T0 + dt*src)) / dt.
    np.testing.assert_allclose(trd, (T1 - (T0 + _DT * rate)) / _DT, rtol=0, atol=1e-15)
    # Zero-flux implicit diffusion conserves each column's heat ...
    col = (trd * dz).sum(-1)[wet]
    assert np.abs(col).max() < 1e-12 * np.abs(trd * dz).sum(-1)[wet].max()
    # ... and is not vacuous: it moves heat, and the total change does not conserve.
    assert np.abs(trd[wet]).max() > 1e-9
    total = (((T1 - T0) / _DT) * dz).sum(-1)[wet]
    assert np.abs(total).min() > 1e3 * np.abs(col).max()


def test_callback_is_read_only_and_fires_under_jit():
    m, st, tke0, src, *_ = _setup()
    got = []
    f = lambda s, cb: m._apply_implicit_vertical_mixing(
        s, _DT, None, tke_old=tke0, return_tke=True, trd_callback=cb)[0].T.data
    with_cb = jax.jit(lambda s: f(s, lambda x, k: got.append((np.asarray(x), np.asarray(k)))))(st)
    without = jax.jit(lambda s: f(s, None))(st)
    np.testing.assert_array_equal(np.asarray(with_cb), np.asarray(without))
    assert len(got) == 1 and got[0][0].shape == st.T.data.shape
    assert got[0][1].shape == st.T.data.shape[:-1] + (st.T.data.shape[-1] - 1,)
    assert np.all(got[0][1] >= 0) and np.any(got[0][1] > 0)


def test_accumulator_mean_empty_window_and_never_fed():
    acc = _core2()._ZdfTrendAccumulator()
    with pytest.raises(SystemExit, match="does not emit"):
        acc.drain(_DT)
    acc(np.full((2, 3), 1.0), np.full((2, 2), 4.0)); acc(np.full((2, 3), 3.0), np.full((2, 2), 6.0))
    out = acc.drain(_DT)
    np.testing.assert_array_equal(out["ttrd_zdf_mean"], np.full((2, 3), 2.0))
    np.testing.assert_array_equal(out["K_trd_mean"], np.full((2, 2), 5.0))
    assert int(out["ttrd_zdf_n_steps"]) == 2
    assert acc.drain(_DT) == {}


def test_box_series_is_the_weighted_profile_per_step():
    w = np.zeros((3, 4)); w[1, 1:3] = [1.0, 3.0]
    acc = _core2()._ZdfTrendAccumulator(w)
    x = np.arange(3 * 4 * 32, dtype=float).reshape(3, 4, 32); k = x[..., :31] * 2.0
    acc(x, k); acc(x + 1.0, k)
    out = acc.drain(_DT)
    want = 0.25 * x[1, 1, :30] + 0.75 * x[1, 2, :30]
    np.testing.assert_allclose(out["ttrd_series"], [want, want + 1.0])
    np.testing.assert_allclose(out["K_series"][0], 2.0 * want[:29])
    assert out["ttrd_series"].shape == (2, 30)


def test_non_tripole_grid_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "mpas", "--trd-accumulate"])
    with pytest.raises(SystemExit, match="tripole only"):
        _core2().main()


def test_column_callback_emits_the_returned_post_solve_state():
    m, st, tke0, src, rate, lm, dz = _setup()
    got = []
    f = lambda s, cb: m._apply_implicit_vertical_mixing(
        s, _DT, None, tke_old=tke0, return_tke=True, surface_tracer_forcing=src,
        col_callback=cb)
    out, tke_out = jax.jit(lambda s: f(s, lambda *a: got.append([np.asarray(x) for x in a])))(st)
    ref = jax.jit(lambda s: f(s, None)[0])(st)
    np.testing.assert_array_equal(np.asarray(out.T.data), np.asarray(ref.T.data))
    assert len(got) == 1
    T, S, u, v, K, A, e = got[0]
    # e is the returned prognostic TKE, and it moved from the seed
    np.testing.assert_array_equal(e, np.asarray(tke_out))
    assert e.shape == K.shape and np.abs(e - np.asarray(tke0)).max() > 1e-8
    np.testing.assert_array_equal(T, np.asarray(out.T.data))
    np.testing.assert_array_equal(S, np.asarray(out.S.data))
    np.testing.assert_array_equal(u, np.asarray(out.u.data))
    np.testing.assert_array_equal(v, np.asarray(out.v.data))
    assert K.shape == A.shape == T.shape[:-1] + (T.shape[-1] - 1,)
    assert np.any(A > 0) and np.any(K > 0)
    # non-vacuous: the solve changed T, so the emitted T is not the input
    assert np.abs(T - np.asarray(st.T.data)).max() > 1e-6


def test_accumulator_columns_keep_selected_cells_every_step():
    acc = _core2()._ZdfTrendAccumulator()
    mask = np.zeros((3, 4), bool); mask[0, 1] = mask[2, 3] = True
    acc.set_columns(mask)
    T = np.arange(3 * 4 * 32, dtype=float).reshape(3, 4, 32); K = T[..., :31]
    for i in range(3):
        acc(T, K); acc.col(T + i, T, T, T, K + i, K, K + 2 * i)
    out = acc.drain(_DT)
    assert out["col_T"].shape == (3, 2, 30) and out["col_K"].shape == (3, 2, 29)
    assert out["col_e"].shape == (3, 2, 29)
    np.testing.assert_array_equal(out["col_e"][2, 1], (K[2, 3, :29] + 4).astype(np.float32))
    np.testing.assert_array_equal(out["col_T"][2, 1], (T[2, 3, :30] + 2).astype(np.float32))
    np.testing.assert_array_equal(out["col_K"][1, 0], (K[0, 1, :29] + 1).astype(np.float32))
    np.testing.assert_array_equal(out["col_j"], [0, 2]); np.testing.assert_array_equal(out["col_i"], [1, 3])
    assert acc.drain(_DT) == {}


def test_columns_without_accumulate_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "tripole",
                                      "--trd-columns", "225", "255", "2"])
    with pytest.raises(SystemExit, match="needs --trd-accumulate"):
        _core2().main()
