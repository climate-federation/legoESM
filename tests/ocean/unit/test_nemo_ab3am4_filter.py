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
    _NEMO_TS_BCK_FLT2,
    nemo_ab3am4_coeff_arrays,
)

_GYRE_FILTER_ALPHA = 0.07


def test_flt2_interior_ssh_weights_are_alpha0_literals():
    """nn_bt_flt=2 (flt2=True): the jn>=3 ssh half-step-back weights are NEMO's
    hard-coded rn_bt_alpha=0 literals 0.614/0.285/0.088/0.013
    (dynspg_ts.F90:1698-1701), NOT the Demange formula — and the AB3 velocity
    weights are unchanged (shared by both filters)."""
    za, zb = nemo_ab3am4_coeff_arrays(10, alpha=0.0, flt2=True)
    np.testing.assert_allclose(np.asarray(zb)[5], _NEMO_TS_BCK_FLT2, atol=1e-12)
    np.testing.assert_allclose(np.asarray(za)[5], _NEMO_AB3_ZA, atol=1e-12)
    # rows still sum to 1 (exact on a constant field)
    np.testing.assert_allclose(np.asarray(zb).sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(za).sum(axis=1), 1.0, atol=1e-12)
    # dissipative and DISTINCT from the nn_bt_flt=3 (alpha=0.07) weights
    _, zb3 = nemo_ab3am4_coeff_arrays(
        10, alpha=_GYRE_FILTER_ALPHA, flt2=False)
    assert not np.allclose(np.asarray(zb)[5], np.asarray(zb3)[5])
    assert float(_NEMO_TS_BCK_FLT2[0]) > 0.5  # forward-weighted (dissipative)


def test_flt2_ll_init_ramp_shared():
    """flt2 keeps the ll_init ramp (nn_bt_flt=2 re-inits every step)."""
    za, zb = nemo_ab3am4_coeff_arrays(4, alpha=0.0, flt2=True)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(
        np.asarray(zb)[1],
        [1.0833333333333, -0.1666666666666, 0.0833333333333, 0.0], atol=1e-12)


def test_coefficient_rows_sum_to_one():
    """Consistency: extrapolation/interpolation exact on a constant field —
    every row of za and zb sums to 1 (NEMO's coefficients do)."""
    za, zb = nemo_ab3am4_coeff_arrays(50, alpha=_GYRE_FILTER_ALPHA)
    np.testing.assert_allclose(np.asarray(za).sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(zb).sum(axis=1), 1.0, atol=1e-12)


def test_interior_coefficients_match_ts_bck_interp_alpha007():
    """The alpha=0.07 AM4 branch (dynspg_ts.F90 ts_bck_interp), NOT the
    alpha==0 published table (0.614/0.285/0.088/0.013)."""
    za, zb = nemo_ab3am4_coeff_arrays(10, alpha=_GYRE_FILTER_ALPHA)
    a = _GYRE_FILTER_ALPHA
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
    za, zb = nemo_ab3am4_coeff_arrays(4, alpha=_GYRE_FILTER_ALPHA)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(za)[1], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(
        np.asarray(zb)[1],
        [1.0833333333333, -0.1666666666666, 0.0833333333333, 0.0], atol=1e-12)


def test_n1_edge_is_forward():
    za, zb = nemo_ab3am4_coeff_arrays(1, alpha=_GYRE_FILTER_ALPHA)
    np.testing.assert_allclose(np.asarray(za)[0], [1.0, 0.0, 0.0], atol=0)
    np.testing.assert_allclose(np.asarray(zb)[0], [1.0, 0.0, 0.0, 0.0], atol=0)


def test_wide_halo_rejected_and_filter_typo_raises():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    assert r.model_config.barotropic.barotropic_time_filter == "nemo_ab3am4"
    assert r.model_config.barotropic.nemo_barotropic_filter_alpha == 0.07
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
    with pytest.raises(ValueError, match="nemo_barotropic_filter_alpha"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(
                barotropic=r.model_config.barotropic._replace(
                    nemo_barotropic_filter_alpha=None)))


def test_ramp_false_gives_full_rows():
    """ramp=False (continuation windows): full AB3/AM4 coefficients from
    substep 0 — no ll_init override."""
    za_r, zb_r = nemo_ab3am4_coeff_arrays(
        4, alpha=_GYRE_FILTER_ALPHA, ramp=True)
    za_f, zb_f = nemo_ab3am4_coeff_arrays(
        4, alpha=_GYRE_FILTER_ALPHA, ramp=False)
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


def test_before_level_seed_none_vs_now_is_byte_identical():
    """Backward-compat guard for the MLF Nbb seed: seeding the override with the
    NOW state (eta_init=state.eta, u_init=state.u, v_init=state.v) must reproduce
    the None-default result bit-for-bit — the U_bar_corr==U_bar / u_corr==u
    collapse in the no-override path (barotropic_latlon_cgrid.py)."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    r, st, (F_eta, F_u, F_v) = _gyre_solver_setup()
    n, dt_e = 12, 100.0
    kw = dict(grid=r.grid, z_coord=r.z_coord, config=r.model_config,
              F_slow_eta=F_eta, F_slow_u=F_u, F_slow_v=F_v,
              add_barotropic_coriolis=False)
    s_default, (Hu0, Hv0) = barotropic_substeps_latlon_cgrid(st, dt_e, n, **kw)
    s_nowseed, (Hu1, Hv1) = barotropic_substeps_latlon_cgrid(
        st, dt_e, n, eta_init=st.eta.data, u_init=st.u.data,
        v_init=st.v.data, **kw)
    for a, b in ((s_default.eta.data, s_nowseed.eta.data),
                 (s_default.u.data, s_nowseed.u.data),
                 (s_default.v.data, s_nowseed.v.data),
                 (Hu0, Hu1), (Hv0, Hv1)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_boxcar_ab3_requires_leapfrog():
    """nemo_boxcar_ab3 (nn_bt_flt=2) is flt=2-faithful only under the MLF
    leap-frog (×2 substep scale + Nbb seed); pairing it with forward_euler is a
    non-NEMO hybrid and must raise."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    mc = r.model_config._replace(
        outer_integrator="forward_euler",
        barotropic=r.model_config.barotropic._replace(
            barotropic_time_filter="nemo_boxcar_ab3"))
    with pytest.raises(ValueError, match="nemo_boxcar_ab3"):
        LatLonCGridOceanModel(r.grid, r.z_coord, mc)


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


def test_nemo_stage_mean_imposition_noop_and_helper():
    """NEMO stprk3_stg:440 zub correction (nemo_stage_mean_imposition):
    (a) default OFF is the legacy path; (b) ON is bit-identical for the GYRE
    card — legoESM's implicit vertical solve has zero-flux BCs and stress/drag
    are applied pre-barotropic, so the depth mean is already conserved (the
    imposition is structurally inert here; it activates only for configs whose
    implicit solve shifts the mean); (c) the depth-mean helper is exact on a
    synthetic column-uniform shift."""
    import jax.numpy as jnp
    import numpy as np

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
        nemo_gyre_wind_forcing,
    )

    r = build_nemo_gyre_recipe()
    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    sf = nemo_gyre_wind_forcing(n_lat, n_lon, 0.0)
    m_off = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    mc_on = r.model_config._replace(
        barotropic=r.model_config.barotropic._replace(
            nemo_stage_mean_imposition=True))
    m_on = LatLonCGridOceanModel(r.grid, r.z_coord, mc_on)
    s_off = m_off.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    s_on = m_on.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    np.testing.assert_array_equal(np.asarray(s_on.u.data),
                                  np.asarray(s_off.u.data))
    np.testing.assert_array_equal(np.asarray(s_on.v.data),
                                  np.asarray(s_off.v.data))

    # helper exactness: a column-uniform shift is recovered exactly
    du = 0.01
    st_shift = st._replace(u=st.u.replace(
        data=(st.u.data + du) * st.u_mask.data[..., None]))
    um0, _ = m_on._fixed_depth_means(st)
    um1, _ = m_on._fixed_depth_means(st_shift)
    wet = np.asarray(st.u_mask.data) > 0
    np.testing.assert_allclose(np.asarray(um1 - um0)[wet], du, rtol=1e-6)  # f32 state


# ---------------------------------------------------------------------------
# nn_bt_flt=2 vvl: mid-step AB3-extrapolated continuity-flux depth
# (dynspg_ts.F90:556-595 — zsshp2_e feeds zhup2_e feeds zhU)
# ---------------------------------------------------------------------------

def _tiny_vvl_basin(H=500.0, n_lat=16, n_lon=24):
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(n_levels=3, H_max=H, dz_surface=50.0, dz_deep=250.0)
    assert not getattr(z, "linear_free_surface", False)  # vvl path under test
    latd = np.asarray(grid.lat2d) * 180.0 / np.pi
    H_bathy = jnp.full((grid.n_lat, grid.n_lon), H)
    land_mask = jnp.asarray(np.where(np.abs(latd) < 75.0, 1.0, 0.0))
    st = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_max=H, H_bathy_override=H_bathy, land_mask_override=land_mask)
    cfg = LatLonCGridOceanConfig(fix_eta_drift=False)
    cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        bebt=0.0, maxvel_barotropic=0.0, barotropic_diffusion_alpha=0.0,
        barotropic_div_damp=0.0, barotropic_local_subcycle_clamp=True,
        barotropic_solver="explicit_substep",
        barotropic_time_filter="nemo_boxcar_ab3",
        nemo_barotropic_filter_alpha=0.0))
    return grid, z, st, cfg


def test_flt2_vvl_uniform_eta_stays_exact():
    """Flat bottom + column-uniform eta offset: the AB3 mid-step ssh
    extrapolation (rows sum to 1) reproduces the SAME uniform eta, the PGF is
    zero, and the state stays exactly at rest through the full subcycle —
    sensitive to any coefficient-row normalisation error in the flux depth."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    grid, z, st, cfg = _tiny_vvl_basin()
    eta0 = 2.0 * st.land_mask.data.astype(jnp.float64)
    st = st._replace(
        eta=st.eta.replace(data=eta0),
        u=st.u.replace(data=st.u.data.astype(jnp.float64)),
        v=st.v.replace(data=st.v.data.astype(jnp.float64)))
    sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
        st, 600.0, 8, grid, z, cfg, add_barotropic_coriolis=False)
    np.testing.assert_array_equal(np.asarray(sn.eta.data), np.asarray(eta0))
    assert float(np.max(np.abs(np.asarray(sn.u.data)))) == 0.0
    assert float(np.max(np.abs(np.asarray(sn.v.data)))) == 0.0
    assert float(np.max(np.abs(np.asarray(Hu)))) == 0.0


def test_flt2_vvl_substep_sequence_matches_nemo_reference():
    """Oracle for the full nn_bt_flt=2 vvl substep composition: a Python
    per-substep reference implementing NEMO's exact sequence
    (dynspg_ts.F90:520-839, DINO-active branch: AB3 velocity + ssh mid-step
    extrapolation feeding the flux depth, forward ssh, ts_bck_interp SPG,
    vector-form velocity update, boxcar accumulation) must reproduce the
    solver bit-tightly.  A large eta/H ratio makes the MID-STEP flux depth
    (vs the substep-start depth) an O(1e-3) effect — asserted non-vacuous
    below, so this test FAILS if the extrapolated depth is dropped."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.ocean.dynamics.barotropic_common import (
        compute_nemo_boxcar_centred_weights,
    )
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        divergence_cgrid,
        gradient_x_cgrid,
        gradient_y_cgrid,
        pad_ns_zero,
    )

    grid, z, st, cfg = _tiny_vvl_basin(H=500.0)
    n_lat, n_lon = st.eta.data.shape
    y, x = np.meshgrid(np.linspace(-1, 1, n_lat), np.linspace(-1, 1, n_lon),
                       indexing="ij")
    eta0 = 50.0 * np.exp(-3.0 * (x**2 + y**2)) * np.asarray(st.land_mask.data)
    st = st._replace(
        eta=st.eta.replace(data=jnp.asarray(eta0, dtype=jnp.float64)),
        u=st.u.replace(data=st.u.data.astype(jnp.float64)),
        v=st.v.replace(data=st.v.data.astype(jnp.float64)))
    n, dt_s = 6, 400.0
    sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
        st, dt_s, n, grid, z, cfg, add_barotropic_coriolis=False)

    # ---- reference: NEMO's substep sequence with shared spatial operators ----
    g = float(cfg.g)
    mask = np.asarray(st.land_mask.data, dtype=np.float64)
    u_mask = np.asarray(st.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(st.v_mask.data, dtype=np.float64)
    H_bathy = float(500.0)
    minw = float(cfg.min_water_column_m)
    eta_floor = minw - H_bathy

    def faces(H_tot):
        H_tot = jnp.asarray(H_tot)
        H_u = jnp.minimum(jnp.roll(H_tot, 1, axis=1), H_tot)
        H_u = jnp.concatenate([H_u, H_u[:, 0:1]], axis=1)
        Hp = pad_ns_zero(H_tot)
        H_v = jnp.minimum(Hp[:-1], Hp[1:])
        H_v = H_v.at[0].set(0.0).at[-1].set(0.0)
        return np.asarray(H_u), np.asarray(H_v)

    w_filter, w_total, w_transport, n_loop = compute_nemo_boxcar_centred_weights(
        n, jnp.float64)
    za, zb = nemo_ab3am4_coeff_arrays(
        n_loop, alpha=0.0, ramp=True, flt2=True)
    za, zb = np.asarray(za), np.asarray(zb)
    w_filter = np.asarray(w_filter)
    w_transport = np.asarray(w_transport)

    eta = np.maximum(np.asarray(eta0, dtype=np.float64), eta_floor) * mask
    U = np.zeros((n_lat, n_lon + 1))
    V = np.zeros((n_lat + 1, n_lon))
    Ub = Ubb = U.copy()
    Vb = Vbb = V.copy()
    etab = etabb = eta.copy()
    eta_sum = np.zeros_like(eta)
    U_sum = np.zeros_like(U)
    V_sum = np.zeros_like(V)
    Hu_sum = np.zeros_like(U)
    for i in range(int(n_loop)):
        H_tot = np.maximum(eta + H_bathy, minw) * mask
        H_u, H_v = faces(H_tot)                       # jn depth (drag; unused)
        U_mid = za[i, 0] * U + za[i, 1] * Ub + za[i, 2] * Ubb
        V_mid = za[i, 0] * V + za[i, 1] * Vb + za[i, 2] * Vbb
        eta_mid = za[i, 0] * eta + za[i, 1] * etab + za[i, 2] * etabb
        H_um, H_vm = faces(np.maximum(eta_mid + H_bathy, minw) * mask)
        flux_u = H_um * U_mid * u_mask                # dynspg_ts:604-609
        flux_v = H_vm * V_mid * v_mask
        Hu_sum = Hu_sum + w_transport[i] * flux_u
        div = np.asarray(divergence_cgrid(
            jnp.asarray(flux_u), jnp.asarray(flux_v), grid,
            u_mask=jnp.asarray(u_mask), v_mask=jnp.asarray(v_mask)))
        eta_new = np.maximum((eta - dt_s * div) * mask, eta_floor) * mask
        eta_pgf = (zb[i, 0] * eta_new + zb[i, 1] * eta
                   + zb[i, 2] * etab + zb[i, 3] * etabb)  # ts_bck_interp
        U_new = (U + dt_s * (-g * np.asarray(
            gradient_x_cgrid(jnp.asarray(eta_pgf), grid)))) * u_mask
        V_new = (V + dt_s * (-g * np.asarray(
            gradient_y_cgrid(jnp.asarray(eta_pgf), grid)))) * v_mask
        eta_sum = eta_sum + w_filter[i] * eta_new     # dynspg_ts:820-836
        U_sum = U_sum + w_filter[i] * U_new
        V_sum = V_sum + w_filter[i] * V_new
        Ubb, Ub, U = Ub, U, U_new                     # dynspg_ts:806-816
        Vbb, Vb, V = Vb, V, V_new
        etabb, etab, eta = etab, eta, eta_new
    wt = float(w_total)
    eta_ref = eta_sum / wt
    U_ref = U_sum / wt

    # atol=5e-8: above the jax(XLA-fused)-vs-numpy float-reassociation noise
    # floor (observed max |diff| ~7e-9 over the 11-substep chain) and 20x
    # below the >1e-6 mid-step-depth signal the non-vacuity guard pins —
    # bit-tight (1e-10) equality across the two execution engines is not
    # achievable and was the only failure mode.
    np.testing.assert_allclose(np.asarray(sn.eta.data), eta_ref,
                               rtol=0, atol=5e-8)
    np.testing.assert_allclose(np.asarray(sn.u.data[..., 0]), U_ref,
                               rtol=0, atol=5e-8)
    np.testing.assert_allclose(np.asarray(Hu), Hu_sum, rtol=0, atol=5e-8)

    # Non-vacuous: dropping the mid-step depth (H_u from eta, NEMO :556-595
    # skipped) must NOT reproduce the solver — guards the gate itself.
    eta2 = np.maximum(np.asarray(eta0, dtype=np.float64), eta_floor) * mask
    U2 = np.zeros_like(U); V2 = np.zeros_like(V)
    U2b = U2bb = U2.copy(); V2b = V2bb = V2.copy()
    eta2b = eta2bb = eta2.copy()
    eta2_sum = np.zeros_like(eta2)
    for i in range(int(n_loop)):
        H_tot = np.maximum(eta2 + H_bathy, minw) * mask
        H_u, H_v = faces(H_tot)
        U_mid = za[i, 0] * U2 + za[i, 1] * U2b + za[i, 2] * U2bb
        V_mid = za[i, 0] * V2 + za[i, 1] * V2b + za[i, 2] * V2bb
        flux_u = H_u * U_mid * u_mask                 # substep-START depth
        flux_v = H_v * V_mid * v_mask
        div = np.asarray(divergence_cgrid(
            jnp.asarray(flux_u), jnp.asarray(flux_v), grid,
            u_mask=jnp.asarray(u_mask), v_mask=jnp.asarray(v_mask)))
        eta_new = np.maximum((eta2 - dt_s * div) * mask, eta_floor) * mask
        eta_pgf = (zb[i, 0] * eta_new + zb[i, 1] * eta2
                   + zb[i, 2] * eta2b + zb[i, 3] * eta2bb)
        U_new = (U2 + dt_s * (-g * np.asarray(
            gradient_x_cgrid(jnp.asarray(eta_pgf), grid)))) * u_mask
        V_new = (V2 + dt_s * (-g * np.asarray(
            gradient_y_cgrid(jnp.asarray(eta_pgf), grid)))) * v_mask
        eta2_sum = eta2_sum + w_filter[i] * eta_new
        U2bb, U2b, U2 = U2b, U2, U_new
        V2bb, V2b, V2 = V2b, V2, V_new
        eta2bb, eta2b, eta2 = eta2b, eta2, eta_new
    # 2e-7: 4x above the 5e-8 match tolerance, ~4.5x below the measured
    # mid-step-depth signal (8.95e-7 on this basin/amplitude/substep count).
    assert float(np.max(np.abs(eta2_sum / wt - eta_ref))) > 2e-7
