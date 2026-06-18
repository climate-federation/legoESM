"""Unit tests for :mod:`legoesm.atmosphere.dynamics.column_les` (the column-LES
orchestration; the manifest-looping CLI lives in ``scripts/run/run_column_les.py``).

The heavy plane-LES run (``run_forced_les``) and ``main`` need real data and are
not unit-tested here; every importable orchestration helper is, with a small
LES resolution + lat-lon grid and the LES run injected as a mock:
build_column_les_setup (grid/coord/forcing/relaxation), extract_gcm_column,
process_column, and run_column_les_pipeline (run + diagnose dispatch).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.dynamics.column_les import (
    ColumnLESConfig,
    ColumnLESSetup,
    build_column_les_setup,
    coefficient_value,
    extract_gcm_column,
    process_column,
    run_column_les_pipeline,
    validate_column_les_config,
)
from legoesm.atmosphere.dynamics.les_regime import (
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.atmosphere.dynamics.les_vertical_mapping import (
    interpolate_column_to_les,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

jax.config.update("jax_enable_x64", True)

# Small LES box so the grid constructs fast: 50 m * 8 = 400 m < 2000 m top.
_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0
)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)
_CONFIG = ColumnLESConfig(regime=_SMALL_REGIME)

_NLEV_GCM = 6


def _gcm_column():
    """A small GCM column whose top (2500 m) exceeds the LES top (2000 m)."""
    gcm_z = jnp.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    gcm_theta = jnp.linspace(300.0, 320.0, _NLEV_GCM)
    ls = ColumnLargeScaleState(
        lat_rad=jnp.deg2rad(20.0),
        T=jnp.linspace(290.0, 230.0, _NLEV_GCM),
        p_full=jnp.linspace(9.5e4, 2.0e4, _NLEV_GCM),
        q_v=jnp.linspace(1e-2, 1e-4, _NLEV_GCM),
        omega=jnp.full((_NLEV_GCM,), 0.02),  # subsidence
        theta_adv=jnp.full((_NLEV_GCM,), -1e-5),
        qv_adv=jnp.full((_NLEV_GCM,), -1e-8),
    )
    return gcm_z, gcm_theta, ls


def test_build_setup_shapes_and_relaxation():
    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=float(jnp.deg2rad(20.0)),
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    assert isinstance(setup, ColumnLESSetup)
    assert setup.regime == "shallow"  # CAPE 200 < 1000
    assert setup.height_coord.n_levels == 8
    # relaxation target/rate on the LES grid.
    assert setup.relax_theta_target.shape == (8,)
    assert setup.relax_rate.shape == (8,)
    # rate zero near the surface, positive at the top.
    z = np.asarray(setup.height_coord.z_full)
    rate = np.asarray(setup.relax_rate)
    assert rate[np.argmin(z)] == pytest.approx(0.0)
    assert rate[np.argmax(z)] > 0.0
    # forcing physics is callable.
    assert callable(setup.forcing_physics)
    # moist IC interpolated onto the LES grid.
    assert setup.q_v_init.shape == (8,)
    assert bool(jnp.all(setup.q_v_init >= 0.0))


def test_forcing_profiles_interpolated_to_les_grid():
    """Nonconstant GCM forcing must be interpolated onto the LES grid before
    make_plane_ls_forcing_physics (which expects nlev_LES profiles)."""
    gcm_z, gcm_theta, ls = _gcm_column()
    # Nonconstant subsidence so interpolation is non-trivial.
    ls = ls._replace(omega=jnp.linspace(0.05, 0.0, _NLEV_GCM))
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    # The forcing physics closes over (nlev_LES,) profiles; the height coord has
    # nlev_LES levels. We can't read the closure directly, but the setup built
    # without a shape error proves the GCM->LES interpolation ran (a raw
    # nlev_gcm profile would mismatch nlev_LES inside the dycore physics).
    assert setup.height_coord.n_levels == 8
    # theta_adv channel likewise interpolates: build a reference and confirm the
    # interpolation helper maps gcm levels -> LES levels (8,).
    interp = interpolate_column_to_les(
        gcm_z, ls.theta_adv, setup.height_coord.z_full)
    assert interp.shape == (8,)


def test_geostrophic_wind_wired_into_les_coriolis():
    """A GCM column with a geostrophic wind sets the LES height-coordinate
    reference wind (u_geo0/v_geo0), which the plane f-plane Coriolis reads as
    f×(V−V_geo); None leaves it unset (f×V), unchanged (iter 29)."""
    gcm_z, gcm_theta, ls = _gcm_column()
    # No geostrophic wind → reference winds stay None.
    setup_none = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    assert setup_none.height_coord.u_geo0 is None
    assert setup_none.height_coord.v_geo0 is None

    # With a geostrophic wind → interpolated onto the LES grid + wired in.
    ls_geo = ls._replace(
        u_geo=jnp.linspace(5.0, 12.0, _NLEV_GCM),
        v_geo=jnp.linspace(-2.0, 1.0, _NLEV_GCM),
    )
    setup_geo = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls_geo, config=_CONFIG,
    )
    assert setup_geo.height_coord.u_geo0 is not None
    assert setup_geo.height_coord.u_geo0.shape == (8,)  # LES nlev
    assert setup_geo.height_coord.v_geo0.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(setup_geo.height_coord.u_geo0)))
    assert bool(jnp.all(jnp.isfinite(setup_geo.height_coord.v_geo0)))


def test_validate_config_rejects_bad_settings():
    validate_column_les_config(ColumnLESConfig())  # defaults OK
    for bad in (
        ColumnLESConfig(relax_width_frac=0.0),
        ColumnLESConfig(relax_width_frac=1.5),
        ColumnLESConfig(relax_tau_s=0.0),
        ColumnLESConfig(p_sfc_Pa=-1.0),
        # clubb_coefficient method without l_mix_max is rejected.
        ColumnLESConfig(diagnosis_method="clubb_coefficient"),
        ColumnLESConfig(diagnosis_method="clubb_coefficient", clubb_l_mix_max=0.0),
        # clubb_coefficient as ONE of the multi methods still needs l_mix_max.
        ColumnLESConfig(diagnosis_methods=("clubb_coefficient", "prandtl_number")),
        ColumnLESConfig(diagnosis_methods=()),   # empty multi list
        # c_eps also needs l_mix_max (single + multi).
        ColumnLESConfig(diagnosis_method="c_eps"),
        ColumnLESConfig(diagnosis_methods=("c_eps", "prandtl_number")),
        # realism thresholds must be None or finite-positive (iter 66).
        ColumnLESConfig(les_realism_theta_drift_K=0.0),
        ColumnLESConfig(les_realism_theta_drift_K=-1.0),
        ColumnLESConfig(les_realism_theta_drift_K=float("nan")),
        ColumnLESConfig(les_realism_wp2_floor=float("inf")),
        ColumnLESConfig(les_realism_q_v_max=0.0),
        ColumnLESConfig(les_realism_q_v_max=-0.01),
        ColumnLESConfig(les_realism_rh_max=0.0),
        ColumnLESConfig(les_realism_rh_max=float("inf")),
    ):
        with pytest.raises(ValueError):
            validate_column_les_config(bad)
    # None (default) + explicit finite-positive realism thresholds validate.
    validate_column_les_config(ColumnLESConfig(
        les_realism_theta_drift_K=5.0, les_realism_wp2_floor=1e-3,
        les_realism_q_v_max=0.05, les_realism_rh_max=1.5))
    # clubb_coefficient WITH a positive l_mix_max validates (single + multi).
    validate_column_les_config(
        ColumnLESConfig(diagnosis_method="clubb_coefficient", clubb_l_mix_max=100.0))
    validate_column_les_config(ColumnLESConfig(
        diagnosis_methods=("clubb_coefficient", "prandtl_number"),
        clubb_l_mix_max=100.0))
    # a multi list WITHOUT clubb_coefficient needs no l_mix_max.
    validate_column_les_config(
        ColumnLESConfig(diagnosis_methods=("prandtl_number", "entrainment")))


def test_coefficient_value_dispatch():
    class _K:
        K = jnp.zeros(7)

    class _Ent:
        w_entrainment = jnp.asarray(0.01)

    class _Ck:
        C_K = jnp.full((7,), 0.3)

    class _Prt:
        Pr_t = jnp.full((7,), 0.8)

    class _CEps:
        C_eps = jnp.full((7,), 0.3)

    assert coefficient_value(_K(), "eddy_diffusivity").shape == (7,)
    assert float(coefficient_value(_Ent(), "entrainment")) == pytest.approx(0.01)
    assert coefficient_value(_Ck(), "clubb_coefficient").shape == (7,)
    assert coefficient_value(_Prt(), "prandtl_number").shape == (7,)
    assert coefficient_value(_CEps(), "c_eps").shape == (7,)
    with pytest.raises(ValueError, match="no coefficient"):
        coefficient_value(_K(), "entrainment")  # wrong pairing
    with pytest.raises(ValueError, match="no coefficient"):
        coefficient_value(_K(), "clubb_coefficient")  # K has no C_K
    with pytest.raises(ValueError, match="no coefficient"):
        coefficient_value(_K(), "prandtl_number")  # K has no Pr_t


def test_build_setup_deep_regime_by_cape():
    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=2500.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    assert setup.regime == "deep"  # CAPE 2500 >= 1000


def test_build_setup_raises_when_les_top_above_column():
    gcm_z, gcm_theta, ls = _gcm_column()
    # Column top only 1500 m < LES top 2000 m -> flat-extrapolation guard.
    short_z = jnp.array([0.0, 500.0, 1000.0, 1200.0, 1400.0, 1500.0])
    with pytest.raises(ValueError, match="exceeds the GCM column top"):
        build_column_les_setup(
            cape_J_kg=200.0, lat_rad=0.3,
            gcm_z=short_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
        )


def test_run_pipeline_runs_and_diagnoses():
    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )

    # Mock LES run: return a synthetic plane state on the setup's grid/coord.
    def fake_run(s):
        return _synthetic_plane_state(s.grid, s.height_coord)

    out = run_column_les_pipeline(setup, fake_run, method="eddy_diffusivity")
    assert out.K.shape == (s_nlev(setup) - 1,)
    # Unknown diagnosis method raises (dispatch hardening).
    with pytest.raises(ValueError, match="Unknown column-LES diagnosis method"):
        run_column_les_pipeline(setup, fake_run, method="bogus")


def test_run_pipeline_realism_gate_invalidates_dead_les():
    """A DEAD LES (rest state, no turbulence) has its diagnosis INVALIDATED by the
    realism gate (default on) so the loop keeps the background; toggling the gate
    off leaves the (vacuous) diagnosis ungated; a turbulent LES stays valid."""
    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG)

    def dead_run(s):                       # zero w ⇒ wp2 ≈ 0 ⇒ not realistic
        st = _synthetic_plane_state(s.grid, s.height_coord)
        return st._replace(w=st.w.replace(data=jnp.zeros_like(st.w.data)))

    # Dead LES + gate on (default) ⇒ the whole diagnosis is invalidated.
    dead = run_column_les_pipeline(setup, dead_run, method="eddy_diffusivity")
    assert not bool(jnp.any(dead.valid))
    # Gate OFF on the SAME dead LES ⇒ the K=0 (zero-flux) diagnosis keeps its own
    # per-level validity at gradient levels — so the realism gate is provably what
    # invalidated the gated run above (not the K-diagnosis itself).
    dead_nogate = run_column_les_pipeline(
        setup, dead_run, method="eddy_diffusivity", gate_realism=False)
    assert bool(jnp.any(dead_nogate.valid))
    # The multi path gates EVERY method from the shared realism flag.
    dead_multi = run_column_les_pipeline(
        setup, dead_run, methods=["clubb_coefficient", "prandtl_number"],
        l_mix_max=100.0)
    assert not bool(jnp.any(dead_multi["clubb_coefficient"].valid))
    assert not bool(jnp.any(dead_multi["prandtl_number"].valid))


def test_run_pipeline_optin_rh_cap_threads_and_invalidates():
    """iter 68: the OPT-IN supersaturation cap threads config → pipeline → gate. The
    synthetic LES carries the (unphysical) uniform q=0.01, grossly supersaturated
    aloft, so it is VALID by default (cap off) but INVALIDATED once realism_rh_max is
    set — proving the rh_max threading reaches column_les_realism."""
    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG)

    def turb_run(s):                       # sheared+turbulent (valid C_K), q=0.04 —
        st = _synthetic_plane_state(s.grid, s.height_coord)   # < q_v_max so default passes
        z = jnp.asarray(s.height_coord.z_full)
        ny, nx = s.grid.ny, s.grid.nx
        u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))  # constant shear
        return st._replace(
            u=st.u.replace(data=u),
            tracers=st.tracers.replace(
                data=st.tracers.data.at[..., 0].set(0.04)))   # ~2x surface q_sat → RH~2

    base = run_column_les_pipeline(
        setup, turb_run, method="clubb_coefficient", l_mix_max=100.0)
    assert bool(jnp.any(base.valid))       # cap off (default) ⇒ valid
    capped = run_column_les_pipeline(
        setup, turb_run, method="clubb_coefficient", l_mix_max=100.0,
        realism_rh_max=1.5)
    assert not bool(jnp.any(capped.valid))  # cap on ⇒ supersaturated → invalid


def test_run_pipeline_diagnose_many_shares_one_les_run():
    """`methods` runs the LES ONCE and diagnoses every method from the same final
    state — the shared spin-off for multi-coefficient correction."""
    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG)

    n_runs = {"n": 0}

    def fake_run(s):
        n_runs["n"] += 1
        return _synthetic_plane_state(s.grid, s.height_coord)

    out = run_column_les_pipeline(
        setup, fake_run,
        methods=["clubb_coefficient", "prandtl_number"], l_mix_max=100.0)
    assert n_runs["n"] == 1                       # ONE LES run for BOTH diagnoses
    assert set(out) == {"clubb_coefficient", "prandtl_number"}
    assert hasattr(out["clubb_coefficient"], "C_K")
    assert hasattr(out["prandtl_number"], "Pr_t")
    # clubb_coefficient in the list still requires l_mix_max (per-method guard).
    with pytest.raises(ValueError, match="requires l_mix_max"):
        run_column_les_pipeline(setup, fake_run, methods=["clubb_coefficient"])
    with pytest.raises(ValueError, match="non-empty"):
        run_column_les_pipeline(setup, fake_run, methods=[])


def s_nlev(setup):
    return setup.height_coord.n_levels


def _synthetic_plane_state(grid, hc):
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        make_rest_state,
    )

    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    # checkerboard w + theta' so the resolved flux is nonzero.
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    return state._replace(
        w=state.w.replace(data=w),
        theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr),
    )


def test_extract_gcm_column():
    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 10.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    gcm_z, gcm_theta, ls = extract_gcm_column(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)),
    )
    assert gcm_z.shape == (nlev,)
    assert gcm_theta.shape == (nlev,)
    assert isinstance(ls, ColumnLargeScaleState)
    # heights are non-negative; potential temperature increases with height
    # (θ = T/exner, exner decreases upward).
    assert bool(jnp.all(gcm_z >= 0.0))


def test_process_column_end_to_end_with_mock_run():
    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 10.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)

    class _Env:
        cape_J_kg = 200.0

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    def fake_run(s):
        return _synthetic_plane_state(s.grid, s.height_coord)

    out = process_column(
        _Rec(), T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        config=_CONFIG, run_les_fn=fake_run,
    )
    assert out.K.shape == (7,)  # LES nlev 8 -> 7 interior interfaces
    assert bool(jnp.all(jnp.isfinite(out.K)))

    # Multi-coefficient: diagnosis_methods → one LES run, a {method: diagnosis} dict.
    multi_cfg = ColumnLESConfig(
        regime=_SMALL_REGIME,
        diagnosis_methods=("clubb_coefficient", "prandtl_number"),
        clubb_l_mix_max=100.0)
    multi = process_column(
        _Rec(), T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        config=multi_cfg, run_les_fn=fake_run,
    )
    assert set(multi) == {"clubb_coefficient", "prandtl_number"}
    assert multi["clubb_coefficient"].C_K.shape == (7,)
    assert multi["prandtl_number"].Pr_t.shape == (7,)


def test_extract_gcm_column_cubed_sphere():
    """The column extractor is grid-agnostic: a (face,i,j) index on a cubed-
    sphere state gathers the RIGHT column (not another face) + builds the
    forcing (iter 28).  Fields are column-unique so a wrong-face gather is
    caught (Codex iter-28)."""
    from legoesm.atmosphere.physics._shared import exner_function
    from legoesm.grids.factory import create_grid

    res, nlev = 8, 6
    grid = create_grid("cubed_sphere", resolution=res)
    sigma = create_sigma_coordinate(nlev)
    f, i, j = 2, 3, 5
    # Column-unique fields: every (face,i,j,k) value is distinct, so the gather
    # MUST reproduce exactly the (2,3,5) column, not face 0 or any neighbour.
    faces = jnp.arange(6.0)[:, None, None, None]
    ii = jnp.arange(res, dtype=jnp.float64)[None, :, None, None]
    jj = jnp.arange(res, dtype=jnp.float64)[None, None, :, None]
    kk = jnp.arange(nlev, dtype=jnp.float64)[None, None, None, :]
    T = 280.0 + 100.0 * faces + 10.0 * ii + jj + 0.5 * kk
    q_v = 5e-3 + 1e-4 * (faces + ii + jj) + 1e-5 * kk  # varies across levels too
    u = jnp.full((6, res, res, nlev), 10.0)
    v = jnp.zeros((6, res, res, nlev))
    p_s = 1.0e5 + 100.0 * faces[..., 0] + 10.0 * ii[..., 0] + jj[..., 0]

    gcm_z, gcm_theta, ls = extract_gcm_column(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=(f, i, j), lat_rad=float(jnp.deg2rad(20.0)),
    )
    assert gcm_z.shape == (nlev,)
    assert isinstance(ls, ColumnLargeScaleState)
    assert bool(jnp.all(jnp.isfinite(gcm_z)))
    assert bool(jnp.all(gcm_z >= 0.0))

    # The gather picked EXACTLY column (2,3,5) — and not face 0.
    np.testing.assert_array_equal(np.asarray(ls.T), np.asarray(T[f, i, j, :]))
    assert not bool(jnp.allclose(ls.T, T[0, i, j, :]))
    np.testing.assert_allclose(np.asarray(ls.q_v), np.asarray(q_v[f, i, j, :]))
    # gcm_theta is the (2,3,5) θ = T/exner(p_full) with that column's p_s.
    p_full_col = p_s[f, i, j] * jnp.asarray(sigma.sigma_full)
    np.testing.assert_allclose(
        np.asarray(gcm_theta),
        np.asarray(T[f, i, j, :] / exner_function(p_full_col)), rtol=1e-12)


def test_process_column_cubed_sphere_with_mock_run():
    """Full per-column pipeline composes on a cubed-sphere GCM state with a
    (face,i,j) record + a mock LES run (iter 28)."""
    from legoesm.grids.factory import create_grid

    res, nlev = 8, 6
    grid = create_grid("cubed_sphere", resolution=res)
    sigma = create_sigma_coordinate(nlev)
    shape = (6, res, res, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 10.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((6, res, res), 1.0e5)

    class _Env:
        cape_J_kg = 200.0

    class _Rec:
        grid_index = (2, 3, 5)  # (face, i, j)
        lat_deg = 20.0
        environment = _Env()

    def fake_run(s):
        return _synthetic_plane_state(s.grid, s.height_coord)

    out = process_column(
        _Rec(), T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        config=_CONFIG, run_les_fn=fake_run,
    )
    assert out.K.shape == (7,)
    assert bool(jnp.all(jnp.isfinite(out.K)))


def test_process_column_real_dycore_integration():
    """MOCK-FREE end-to-end: extract -> setup -> the REAL plane-NH dycore
    (run_forced_les, a few real steps with the large-scale forcing + top
    relaxation) -> diagnose. Validates the run path the unit tests mock out
    (the compressible-Euler plane LES actually runs and stays finite)."""
    from legoesm.atmosphere.dynamics.column_les import (
        build_column_les_setup,
        run_forced_les,
    )
    from legoesm.atmosphere.dynamics.column_les_diagnosis import (
        diagnose_column_coefficient,
    )

    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 8.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)

    gcm_z, gcm_theta, ls = extract_gcm_column(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)),
    )
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=float(jnp.deg2rad(20.0)),
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    # dx=50 m, n_acoustic_substeps=6, c≈340 ⇒ acoustic CFL = 340·0.5/(6·50) ≈
    # 0.57 < 1; a few steps from rest + a small θ' seed stay finite.
    final = run_forced_les(setup, dt_s=0.5, n_steps=3)

    # The REAL dycore ran and stayed numerically stable (no blow-up): every
    # prognostic finite, |w| bounded, moisture carried (tracer slot 0 present).
    for arr in (final.u.data, final.v.data, final.w.data,
                final.theta_prime.data, final.tracers.data):
        assert bool(jnp.all(jnp.isfinite(arr)))
    assert float(jnp.max(jnp.abs(final.w.data))) < 50.0
    assert final.tracers.data.shape[-1] == 1  # q_v seeded

    # The diagnosis runs on the real LES state and returns a finite K profile.
    out = diagnose_column_coefficient(final, setup.height_coord,
                                      method="eddy_diffusivity")
    assert out.K.shape == (7,)  # LES nlev 8 -> 7 interior interfaces
    assert bool(jnp.all(jnp.isfinite(out.K)))
