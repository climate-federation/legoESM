"""Direct unit gates for the NEMO nn_bt_flt=3 barotropic scheme
(barotropic_time_filter="nemo_ab3am4"): the AB3/AM4 coefficient arrays
(ts_bck_interp transcription), the ll_init ramp, edge cases, and the
validator guards."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _NEMO_AB3_ZA,
    _NEMO_BT_ALPHA,
    nemo_ab3am4_coeff_arrays,
)


def test_coefficient_rows_sum_to_one():
    """Consistency: extrapolation/interpolation exact on a constant field —
    every row of za and zb sums to 1 (NEMO's coefficients do)."""
    za, zb = nemo_ab3am4_coeff_arrays(50)
    np.testing.assert_allclose(np.asarray(za).sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(zb).sum(axis=1), 1.0, atol=1e-12)


def test_interior_coefficients_match_ts_bck_interp_alpha007():
    """The alpha=0.07 AM4 branch (dynspg_ts.F90 ts_bck_interp), NOT the
    alpha==0 published table (0.614/0.285/0.088/0.013)."""
    za, zb = nemo_ab3am4_coeff_arrays(10)
    a = _NEMO_BT_ALPHA
    eps = 0.00976186 - 0.13451357 * a
    gam = 0.08344500 - 0.51358400 * a
    zb0 = 0.5 + gam + 2.0 * a + 2.0 * eps
    np.testing.assert_allclose(
        np.asarray(zb)[5], [zb0, 1.0 - zb0 - gam - eps, gam, eps], atol=1e-12)
    np.testing.assert_allclose(np.asarray(za)[5], _NEMO_AB3_ZA, atol=1e-12)
    # dissipative: forward-weighted interpolation
    assert zb0 > 0.5


def test_ll_init_ramp_rows():
    """Per-window ramp: substep 0 forward/FB, substep 1 forward/AB2-AM3
    (dynspg_ts:536-543 + ts_bck_interp jn==1/2)."""
    za, zb = nemo_ab3am4_coeff_arrays(4)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(za)[1], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(
        np.asarray(zb)[1],
        [1.0833333333333, -0.1666666666666, 0.0833333333333, 0.0], atol=1e-12)


def test_n1_edge_is_forward():
    za, zb = nemo_ab3am4_coeff_arrays(1)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)


def test_wide_halo_rejected_and_filter_typo_raises():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    assert r.model_config.barotropic.barotropic_time_filter == "nemo_ab3am4"
    with pytest.raises(ValueError, match="wide_halo"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(
                barotropic=r.model_config.barotropic._replace(
                    barotropic_wide_halo=True)))
    with pytest.raises(ValueError, match="barotropic_time_filter"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(
                barotropic=r.model_config.barotropic._replace(
                    barotropic_time_filter="typo")))


def test_ramp_false_gives_full_rows():
    """ramp=False (continuation windows): full AB3/AM4 coefficients from
    substep 0 — no ll_init override."""
    za_r, zb_r = nemo_ab3am4_coeff_arrays(4, ramp=True)
    za_f, zb_f = nemo_ab3am4_coeff_arrays(4, ramp=False)
    np.testing.assert_allclose(np.asarray(za_f)[0], _NEMO_AB3_ZA, atol=1e-12)
    np.testing.assert_allclose(np.asarray(za_f)[1], _NEMO_AB3_ZA, atol=1e-12)
    np.testing.assert_allclose(np.asarray(zb_f)[0], np.asarray(zb_f)[2], atol=0)
    # ramp arrays differ on exactly the first two rows
    np.testing.assert_allclose(np.asarray(za_r)[2:], np.asarray(za_f)[2:], atol=0)
    assert not np.allclose(np.asarray(zb_r)[:2], np.asarray(zb_f)[:2])


def _gyre_solver_setup():
    """Recipe grid/config + a nonzero (eta, u, v) rest-perturbed state."""
    import jax.numpy as jnp

    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    nlev = st.T.data.shape[2]
    y, x = np.meshgrid(np.linspace(-1, 1, n_lat), np.linspace(-1, 1, n_lon),
                       indexing="ij")
    eta0 = 0.05 * np.exp(-4.0 * (x**2 + y**2)) * np.asarray(st.land_mask.data)
    rng = np.random.default_rng(7)
    u0 = 0.01 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v0 = 0.01 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    # float64 state: the continuation gate asserts machine-precision
    # equality; the recipe's f32 default would put roundoff at ~2e-8.
    st = st._replace(
        eta=st.eta.replace(data=jnp.asarray(eta0, dtype=jnp.float64)),
        u=st.u.replace(data=jnp.asarray(u0, dtype=jnp.float64)
                       * st.u_mask.data[..., None]),
        v=st.v.replace(data=jnp.asarray(v0, dtype=jnp.float64)
                       * st.v_mask.data[..., None]),
    )
    # fixed slow forcing so consecutive windows see identical RHS
    F_eta = jnp.zeros((n_lat, n_lon))
    F_u = jnp.full((n_lat, n_lon + 1), 1.0e-6) * st.u_mask.data
    F_v = jnp.full((n_lat + 1, n_lon), -1.0e-6) * st.v_mask.data
    return r, st, (F_eta, F_u, F_v)


def test_cross_window_carry_equals_continuous_run():
    """THE continuation gate (NEMO dynspg_ts ll_init=F): two n-substep windows
    with carried bt_hist must reproduce one continuous 2n-substep window
    (same substep dt, same slow forcing) to machine precision — the AB3/AM4
    series is one unbroken time integration across the window boundary."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    r, st, (F_eta, F_u, F_v) = _gyre_solver_setup()
    n = 12
    dt_e = 100.0
    kw = dict(grid=r.grid, z_coord=r.z_coord, config=r.model_config,
              F_slow_eta=F_eta, F_slow_u=F_u, F_slow_v=F_v,
              add_barotropic_coriolis=False)

    assert st.bt_hist is None  # cold start
    s1, _ = barotropic_substeps_latlon_cgrid(st, dt_e, n, **kw)
    assert s1.bt_hist is not None and len(s1.bt_hist) == 6
    s2, _ = barotropic_substeps_latlon_cgrid(s1, dt_e, n, **kw)

    sc, _ = barotropic_substeps_latlon_cgrid(st, dt_e, 2 * n, **kw)

    np.testing.assert_allclose(np.asarray(s2.eta.data), np.asarray(sc.eta.data),
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(s2.u.data), np.asarray(sc.u.data),
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(s2.v.data), np.asarray(sc.v.data),
                               rtol=0, atol=1e-13)
    # histories at the end must agree too (the NEXT window's inputs)
    for a, b in zip(s2.bt_hist, sc.bt_hist):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=0, atol=1e-13)


def test_cross_window_carry_nonvacuous():
    """The carry changes window 2 (vs re-ramping every window): a genuinely
    different second window — guards against the carry being silently
    dropped."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    r, st, (F_eta, F_u, F_v) = _gyre_solver_setup()
    n = 12
    dt_e = 100.0
    kw = dict(grid=r.grid, z_coord=r.z_coord, config=r.model_config,
              F_slow_eta=F_eta, F_slow_u=F_u, F_slow_v=F_v,
              add_barotropic_coriolis=False)

    s1, _ = barotropic_substeps_latlon_cgrid(st, dt_e, n, **kw)
    s2_carried, _ = barotropic_substeps_latlon_cgrid(s1, dt_e, n, **kw)
    s2_ramped, _ = barotropic_substeps_latlon_cgrid(
        s1._replace(bt_hist=None), dt_e, n, **kw)
    assert float(np.max(np.abs(
        np.asarray(s2_carried.eta.data) - np.asarray(s2_ramped.eta.data)))) > 0.0


def test_model_step_populates_and_carries_bt_hist():
    """Production path: LatLonCGridOceanModel.step populates bt_hist on the
    first step and carries/updates it on the second."""
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
        nemo_gyre_wind_forcing,
    )

    r = build_nemo_gyre_recipe()
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    assert st.bt_hist is None
    s1 = model.step(st, dt=_NEMO_GYRE_DT_S,
                    surface_forcing=nemo_gyre_wind_forcing(n_lat, n_lon, 0.0))
    assert s1.bt_hist is not None and len(s1.bt_hist) == 6
    s2 = model.step(s1, dt=_NEMO_GYRE_DT_S,
                    surface_forcing=nemo_gyre_wind_forcing(
                        n_lat, n_lon, _NEMO_GYRE_DT_S))
    assert s2.bt_hist is not None
    # the histories actually advanced
    assert float(jnp.max(jnp.abs(s2.bt_hist[0] - s1.bt_hist[0]))) > 0.0
    for h in s2.bt_hist:
        assert bool(jnp.all(jnp.isfinite(h)))
