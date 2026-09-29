"""Unit tests for :mod:`legoesm.atmosphere.dynamics.les.column_les` (the column-LES
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
from legoesm.atmosphere.forcing.idealized.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.dynamics.les.column_les import (
    ColumnLESConfig,
    ColumnLESSetup,
    build_column_les_setup,
    coefficient_value,
    extract_gcm_column,
    process_column,
    run_column_les_pipeline,
    validate_column_les_config,
)
from legoesm.atmosphere.dynamics.les.les_regime import (
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.atmosphere.dynamics.les.les_vertical_mapping import (
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


def test_relaxation_target_equals_theta_ref_so_sponge_damps_perturbation():
    """SPONGE CONSISTENCY (run_forced_les correctness): the top-relaxation target is the
    SAME GCM-θ interpolation onto hc.z_full as hc.theta_ref, so the sponge damps the
    perturbation toward the GCM reference (tend = -rate·θ'), NOT toward a divergent
    target (which would inject a SPURIOUS tendency -rate·(θ_ref-target) in the
    relaxation layer and contaminate the LES near the top). gcm_theta varies (300→320),
    so target==theta_ref is a non-trivial profile match, and the damping assertion would
    FAIL if a future change built the target or theta_ref differently."""
    from legoesm.atmosphere.dynamics.les.les_vertical_mapping import relaxation_tendency

    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=float(jnp.deg2rad(20.0)),
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    target = np.asarray(setup.relax_theta_target)
    theta_ref = np.asarray(setup.height_coord.theta_ref)
    # theta_ref IS the GCM-θ interpolation onto hc.z_full — pin it to an INDEPENDENT
    # interpolation so a coordinated refactor to a DIFFERENT-but-mutually-equal profile
    # (which the target==theta_ref check alone would miss) still fails (Codex).
    expected = np.asarray(
        interpolate_column_to_les(gcm_z, gcm_theta, setup.height_coord.z_full))
    np.testing.assert_allclose(theta_ref, expected, rtol=1e-12, atol=1e-12)
    # The relaxation target is the SAME interpolation of gcm_theta onto hc.z_full.
    np.testing.assert_allclose(target, theta_ref, rtol=1e-12, atol=1e-12)
    assert not np.allclose(theta_ref, theta_ref[0])   # non-vacuous: θ_ref VARIES with z
    # Downstream: target==theta_ref ⇒ a uniform θ' is damped toward 0 (tend = -rate·θ'),
    # with NO spurious offset. A target≠theta_ref mismatch would add -rate·(θ_ref-target).
    theta_prime = 0.7
    tend = relaxation_tendency(
        jnp.asarray(theta_ref) + theta_prime, setup.relax_theta_target, setup.relax_rate)
    np.testing.assert_allclose(
        np.asarray(tend), -np.asarray(setup.relax_rate) * theta_prime,
        rtol=1e-12, atol=1e-12)


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


def test_build_setup_rejects_unsupported_T_s_surface_bc():
    """prescribe='T_s' (surface-TEMPERATURE BC) still fails LOUD — the forced-LES path
    has no bulk-flux closure C_H·|U|·(θ_sfc−θ_1) to convert T_s to a kinematic flux yet
    (iter 151 left that the documented next channel). prescribe='fluxes' is now
    supported (see test_build_setup_applies_prescribed_surface_fluxes); 'none' builds."""
    gcm_z, gcm_theta, ls = _gcm_column()
    with pytest.raises(ValueError, match="does NOT yet apply"):
        build_column_les_setup(
            cape_J_kg=200.0, lat_rad=0.3, gcm_z=gcm_z, gcm_theta=gcm_theta,
            ls_state=ls._replace(prescribe="T_s", T_s=300.0), config=_CONFIG,
        )
    # prescribe='none' (the surface-flux-free design) still builds normally.
    ok = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG,
    )
    assert isinstance(ok, ColumnLESSetup)


def test_surface_kinematic_flux_tendency_helper():
    """The surface-flux helper puts a POSITIVE-upward kinematic flux on the SURFACE
    (last) cell only, mass-weighted, zero aloft, and is AD-safe."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.forcing.plane_large_scale_forcing import (
        surface_kinematic_flux_tendency,
    )
    from legoesm.grids.vertical import create_stretched_height_coordinate

    hc = create_stretched_height_coordinate(6, H=2000.0, dz_sfc=50.0)
    flux = 0.05
    tend = np.asarray(surface_kinematic_flux_tendency(flux, hc))
    assert tend.shape == (6,)
    np.testing.assert_allclose(tend[:-1], 0.0)         # interior cells untouched
    rho_w = float(np.asarray(hc.rho_ref_half)[-1])
    rho = float(np.asarray(hc.rho_ref)[-1])
    dz = float(np.asarray(hc.dz)[-1])
    np.testing.assert_allclose(tend[-1], flux * rho_w / rho / dz, rtol=1e-12)
    assert tend[-1] > 0.0                              # upward flux ⇒ positive (warming)
    g = jax.grad(lambda f: jnp.sum(surface_kinematic_flux_tendency(f, hc)))(0.05)
    assert np.isfinite(g) and g > 0.0                  # linear in flux, AD-safe


def test_surface_flux_tendency_column_budget():
    """SOURCE BUDGET: the surface flux adds EXACTLY its surface mass flux to the column.
    The mass-weighted column integral of the surface-flux tendency
    ``Σ_k tend_k·ρ_ref[k]·dz[k]`` must equal ``flux_s·ρ_w_sfc`` (the prescribed kinematic
    flux × the surface-interface density) — the conservation identity that makes this a
    correct boundary SOURCE (it is NOT internally conservative; it injects the surface
    flux's worth of heat/moisture, no more, no less, and only at the surface cell)."""
    import numpy as np
    from legoesm.atmosphere.forcing.plane_large_scale_forcing import (
        surface_kinematic_flux_tendency,
    )
    from legoesm.grids.vertical import create_stretched_height_coordinate

    hc = create_stretched_height_coordinate(7, H=3000.0, dz_sfc=40.0)
    rho = np.asarray(hc.rho_ref)
    dz = np.asarray(hc.dz)
    rho_w_sfc = float(np.asarray(hc.rho_ref_half)[-1])
    for flux in (0.05, -0.02, 3e-5):                   # +up, −down (cooling), tiny moist
        tend = np.asarray(surface_kinematic_flux_tendency(flux, hc))
        col_integral = float(np.sum(tend * rho * dz))  # mass-weighted column source
        np.testing.assert_allclose(col_integral, flux * rho_w_sfc, rtol=1e-12)


def test_build_setup_applies_prescribed_surface_fluxes():
    """prescribe='fluxes' now BUILDS (iter 151) and threads the surface kinematic θ/q_v
    fluxes into the plane forcing physics, which injects them on the SURFACE (last) cell
    ONLY. Isolated by DIFFERENCING the flux vs no-flux setups (same ls otherwise, so the
    subsidence/advection channels cancel): the difference is the surface-flux source —
    zero at every interior level, equal to the helper value at the surface cell."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import make_rest_state
    from legoesm.atmosphere.forcing.plane_large_scale_forcing import (
        surface_kinematic_flux_tendency,
    )

    gcm_z, gcm_theta, ls = _gcm_column()
    wth, wqv = 0.05, 2e-5
    common = dict(cape_J_kg=200.0, lat_rad=0.3, gcm_z=gcm_z, gcm_theta=gcm_theta,
                  config=_CONFIG)
    setup_none = build_column_les_setup(ls_state=ls, **common)
    setup_flux = build_column_les_setup(
        ls_state=ls._replace(prescribe="fluxes", w_th_s=wth, w_qv_s=wqv), **common)
    assert isinstance(setup_flux, ColumnLESSetup)
    grid, hc = setup_flux.grid, setup_flux.height_coord
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    # Moist state with a q_v tracer slot (as run_forced_les builds it) so the surface
    # q_v flux has a tracer to act on (a dry rest state has n_tracers=0).
    ny, nx, nlev0 = grid.ny, grid.nx, hc.n_levels
    tracers = jnp.zeros((ny, nx, nlev0, 1), dtype=jnp.float64).at[..., 0].set(5e-3)
    rest = rest._replace(tracers=rest.tracers.replace(data=tracers))
    t_none = setup_none.forcing_physics(rest, grid, hc, None)
    t_flux = setup_flux.forcing_physics(rest, grid, hc, None)
    nlev = hc.n_levels
    d_theta = np.asarray(t_flux.dtheta_prime_dt.data - t_none.dtheta_prime_dt.data)
    d_qv = np.asarray((t_flux.dtracers_dt.data - t_none.dtracers_dt.data)[..., 0])
    exp_th = float(np.asarray(surface_kinematic_flux_tendency(wth, hc))[-1])
    exp_qv = float(np.asarray(surface_kinematic_flux_tendency(wqv, hc))[-1])
    assert exp_th > 0.0 and exp_qv > 0.0               # upward flux ⇒ warming/moistening
    np.testing.assert_allclose(d_theta[:, :, :nlev - 1], 0.0, atol=1e-14)   # interior: 0
    np.testing.assert_allclose(d_qv[:, :, :nlev - 1], 0.0, atol=1e-20)
    np.testing.assert_allclose(d_theta[:, :, nlev - 1], exp_th, rtol=1e-10)  # surface only
    np.testing.assert_allclose(d_qv[:, :, nlev - 1], exp_qv, rtol=1e-10)
    # The surface flux is a SCALAR (θ/q_v) source — it must NOT leak into momentum (u,v)
    # or w; their tendencies are identical with and without the flux.
    np.testing.assert_array_equal(np.asarray(t_flux.du_dt.data), np.asarray(t_none.du_dt.data))
    np.testing.assert_array_equal(np.asarray(t_flux.dv_dt.data), np.asarray(t_none.dv_dt.data))
    np.testing.assert_array_equal(np.asarray(t_flux.dw_dt.data), np.asarray(t_none.dw_dt.data))


def test_build_setup_rejects_nonfinite_surface_flux():
    """A NON-FINITE prescribed surface flux fails LOUD at setup (validate_forcing only
    checks the callables are present, not their VALUES) — a NaN/inf flux would poison
    the LES surface cell and blow up the run; catch it before that."""
    gcm_z, gcm_theta, ls = _gcm_column()
    for bad in (ls._replace(prescribe="fluxes", w_th_s=float("nan"), w_qv_s=2e-5),
                ls._replace(prescribe="fluxes", w_th_s=0.05, w_qv_s=float("inf"))):
        with pytest.raises(ValueError, match="non-finite"):
            build_column_les_setup(
                cape_J_kg=200.0, lat_rad=0.3,
                gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=bad, config=_CONFIG,
            )


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")  # iter 211: no f64->f32 scatter
def test_prescribed_surface_flux_warms_les_surface_real_dycore():
    """END-TO-END in the REAL plane dycore: prescribed POSITIVE surface θ AND q_v fluxes
    WARM + MOISTEN the LES SURFACE cell vs an identical no-flux run (same fixed θ' seed,
    PRNGKey(0)), and the run stays FINITE — the iter-151 injection does the right thing
    through the actual compressible-Euler integration, not just the forcing tendency (a
    wrong-cell or wrong-sign bug would fail: the surface would not warm/moisten). Heavy
    (real LES, two dycore compiles) ⇒ slow."""
    import numpy as np
    from legoesm.atmosphere.dynamics.les.column_les import run_forced_les

    gcm_z, gcm_theta, ls = _gcm_column()
    common = dict(cape_J_kg=200.0, lat_rad=0.3, gcm_z=gcm_z, gcm_theta=gcm_theta,
                  config=_CONFIG)
    s_none = build_column_les_setup(ls_state=ls, **common)
    s_flux = build_column_les_setup(
        ls_state=ls._replace(prescribe="fluxes", w_th_s=0.5, w_qv_s=2e-5), **common)
    fin_none = run_forced_les(s_none, dt_s=0.1, n_steps=20)
    fin_flux = run_forced_les(s_flux, dt_s=0.1, n_steps=20)
    tp_none = np.asarray(fin_none.theta_prime.data)
    tp_flux = np.asarray(fin_flux.theta_prime.data)
    qv_none = np.asarray(fin_none.tracers.data)[:, :, -1, 0]   # surface cell q_v (slot 0)
    qv_flux = np.asarray(fin_flux.tracers.data)[:, :, -1, 0]
    assert np.all(np.isfinite(tp_none)) and np.all(np.isfinite(tp_flux))  # no blow-up
    assert np.all(np.isfinite(qv_none)) and np.all(np.isfinite(qv_flux))
    # Surface = the LAST level (top-to-bottom storage); domain-mean there.
    assert float(np.mean(tp_flux[:, :, -1])) > float(np.mean(tp_none[:, :, -1])) + 1e-3
    # The q_v flux MOISTENS the surface (signal ~8e-7 kg/kg; margin well below it).
    assert float(np.mean(qv_flux)) > float(np.mean(qv_none)) + 1e-8


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


def test_realism_gate_defaults_on_so_production_is_protected():
    """The LES realism TRUST gate defaults ON in ColumnLESConfig. The production campaign
    builds ``ColumnLESConfig()`` (the campaign CLI exposes NO gate-disable flag, verified
    iter 305), so the 'parameters estimated from LES' quality control is structurally
    always-on — a dead / blown-up / drifted LES never injects a meaningless coefficient into
    a multi-day HPC run. Lock the DEFAULT: a regression flipping it to False would silently
    disable that control in production, and no other test would catch it (the gate-behaviour
    tests above pass ``gate_realism`` EXPLICITLY, and ``run_column_les_pipeline``'s own arg
    default is a SEPARATE default from the config field the campaign actually uses)."""
    assert ColumnLESConfig().gate_les_realism is True


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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    # SPATIALLY-VARYING p_s with a DISTINCT (4,8) column value (0.9e5 vs 1.0e5
    # elsewhere) — so the θ check below proves the col_index gather selected (4,8),
    # not just that the conversion is right (a uniform p_s would not distinguish).
    p_s = jnp.full((n_lat, n_lon), 1.0e5).at[4, 8].set(0.9e5)
    gcm_z, gcm_theta, ls = extract_gcm_column(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)),
    )
    assert gcm_z.shape == (nlev,)
    assert gcm_theta.shape == (nlev,)
    assert isinstance(ls, ColumnLargeScaleState)
    assert bool(jnp.all(gcm_z >= 0.0))
    # gcm_z is a STRICTLY MONOTONIC hydrostatic height profile stored TOP-DOWN (index
    # 0 = highest level) — the LES interpolation + the build_top_relaxation
    # top-within-column guard rely on a clean monotonic column (a kink would mean a
    # bad height integral).
    assert bool(jnp.all(jnp.diff(gcm_z) < 0.0))        # strictly descending in index
    # gcm_z magnitude sanity: for p_top ≈ 0.0925·0.9e5 ≈ 83 hPa the column top is tens
    # of km — catches a GROSS hydrostatic-integral error (wrong R_d / g / sign) without
    # brittly replicating the discretized scheme (the per-layer Δz differs ~18% from the
    # simple isothermal form, so a tight analytic match is not meaningful here).
    assert 8_000.0 < float(gcm_z[0]) < 40_000.0
    # gcm_theta is θ = T/exner(p_full) for the (4,8) column's p_s — LOCK it to an
    # INDEPENDENT analytic value (a wrong exner/p_full conversion, OR a wrong column
    # index, fails); θ INCREASES with height (exner decreases upward ⇒ descending in
    # the top-down index).
    from legoesm.atmosphere.physics._shared import exner_function
    p_full_48 = 0.9e5 * jnp.asarray(sigma.sigma_full)         # the (4,8) column's p_full
    np.testing.assert_allclose(
        np.asarray(gcm_theta), np.asarray(280.0 / exner_function(p_full_48)), rtol=1e-12)
    # A WRONG column (p_s=1.0e5 elsewhere) would give a DIFFERENT θ — proves the gather.
    wrong_col = np.asarray(280.0 / exner_function(1.0e5 * jnp.asarray(sigma.sigma_full)))
    assert not np.allclose(np.asarray(gcm_theta), wrong_col)
    assert bool(jnp.all(jnp.diff(gcm_theta) < 0.0))    # θ increases with height


def test_column_surface_kinematic_fluxes_sign_and_reuse():
    """The surface-flux helper REUSES the GCM bulk scheme (no new tunables) and converts
    its W/m² fluxes to the kinematic θ/q_v fluxes the iter-151 LES BC consumes.  Locks: a
    warm SST warms+moistens the surface air (w'θ'_s>0, w'q'_s>0); the result EQUALS
    compute_surface_fluxes converted directly (so it shares the GCM's flux, not a
    re-derived one); a COLD SST flips the sign (surface cools — non-vacuity)."""
    from legoesm.atmosphere.dynamics.les.column_les import column_surface_kinematic_fluxes
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        SurfaceLayerConfig,
        compute_surface_fluxes,
    )
    from legoesm.thermo import saturation_mixing_ratio

    from legoesm import constants

    nlev = 6
    T = jnp.linspace(240.0, 295.0, nlev)        # surface-last (p ascending)  # noqa: N806
    q = jnp.linspace(1e-4, 1.2e-2, nlev)
    u = jnp.full((nlev,), 5.0)
    v = jnp.zeros((nlev,))
    p_full = jnp.linspace(2.0e4, 1.0e5, nlev)   # surface = highest p = last index
    p_s = jnp.array(1.0e5)
    sst = jnp.array(300.0)                       # warm SST > T_1=295
    w_th, w_qv = column_surface_kinematic_fluxes(
        T_col=T, q_v_col=q, u_col=u, v_col=v, p_full_col=p_full, sst_K=sst, p_s=p_s)
    assert float(w_th) > 0.0 and float(w_qv) > 0.0

    # Reuse-equivalence: exactly compute_surface_fluxes converted (NOT a re-derived bulk flux).
    cfg = SurfaceLayerConfig()
    rho = p_full[-1] / (constants.R_d * virtual_temperature(T[-1], q[-1]))
    q_sfc = saturation_mixing_ratio(sst, p_s)
    _, _, sh, lh, _ = compute_surface_fluxes(
        jnp.atleast_1d(u[-1]), jnp.atleast_1d(v[-1]), jnp.atleast_1d(T[-1]),
        jnp.atleast_1d(q[-1]), jnp.atleast_1d(sst), jnp.atleast_1d(q_sfc),
        jnp.atleast_1d(rho), cfg)
    exner_inv = (constants.p_ref / p_s) ** constants.kappa
    # Water: the bulk law charged Kirchhoff L_v(SST), so the inverse uses it too.
    from legoesm.thermo import latent_heat_vaporization
    evap = lh[0] / latent_heat_vaporization(sst)
    np.testing.assert_allclose(float(w_qv), float(evap / rho), rtol=1e-12)
    # Heat: the moist-enthalpy correction lh - L_v * E rides on the heat BC
    # (the LES credits vapour at the constant L_v); ~3 % of lh at 300 K, negative.
    corr = lh[0] - constants.L_v * evap
    assert float(corr) < 0.0 and float(jnp.abs(corr)) > 0.02 * float(lh[0])
    np.testing.assert_allclose(
        float(w_th), float((sh[0] + corr) / (rho * constants.c_pd) * exner_inv), rtol=1e-12)

    # COLD SST (SST < T_1) ⇒ surface COOLS the air ⇒ w'θ'_s < 0 (sign flips) — non-vacuity.
    w_th_cold, _ = column_surface_kinematic_fluxes(
        T_col=T, q_v_col=q, u_col=u, v_col=v, p_full_col=p_full,
        sst_K=jnp.array(250.0), p_s=p_s)
    assert float(w_th_cold) < 0.0


def test_column_surface_kinematic_fluxes_independent_analytic():
    """Adversarial-review lock (iter 376): the kinematic fluxes equal the textbook
    constant-Ch bulk closed form computed INDEPENDENTLY of ``compute_surface_fluxes``
    — so a regression in that function's argument ORDER or internals is caught here,
    which the reuse-equivalence test cannot do (it routes the expected value through
    the SAME function, so both would move together).  Also independently pins (a) the
    rho CANCELLATION — the closed form is rho-FREE — and (b) the surface-exner
    DIRECTION: ``p_s < p_ref`` (high terrain) AMPLIFIES w'θ' (a classic
    inverted-exner bug would damp it).  Uses ``Ch_neutral`` from the config (not a
    hardcoded coefficient) and the shared saturation helper (no re-derivation)."""
    from legoesm.atmosphere.dynamics.les.column_les import (
        column_surface_kinematic_fluxes,
    )
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.thermo import saturation_mixing_ratio

    from legoesm import constants

    nlev = 4
    T_col = jnp.array([240.0, 260.0, 275.0, 285.0])             # noqa: N806
    q_col = jnp.full((nlev,), 6e-3)
    u_col = jnp.full((nlev,), 6.0)
    v_col = jnp.full((nlev,), 3.0)
    p_full = jnp.array([2.0e4, 5.0e4, 7.0e4, 9.0e4])           # surface = last (max p)
    sst = jnp.asarray(292.0)                                    # warmer than T_1 = 285
    p_s = jnp.asarray(0.7 * constants.p_ref)                    # high terrain ⇒ exner > 1

    w_th, w_qv = column_surface_kinematic_fluxes(
        T_col=T_col, q_v_col=q_col, u_col=u_col, v_col=v_col,
        p_full_col=p_full, sst_K=sst, p_s=p_s)

    # Independent textbook closed form (default bulk_scheme="constant"):
    #   w'θ'_s = Ch·|U|·(SST−T₁)·(p_ref/p_s)^κ ,  w'q'_s = Ch·|U|·(q_sat(SST)−q₁) .
    # The ρ₁ in shflx cancels the ρ₁ in the kinematic conversion ⇒ rho-FREE here.
    ch = SurfaceLayerConfig().Ch_neutral
    s = int(jnp.argmax(p_full))
    wind = float(jnp.sqrt(u_col[s] ** 2 + v_col[s] ** 2 + 1e-4))  # production floor
    exner = float((constants.p_ref / p_s) ** constants.kappa)
    q_sfc = float(saturation_mixing_ratio(sst, p_s))
    w_qv_expected = ch * wind * (q_sfc - float(q_col[s]))
    # The heat BC carries the moist-enthalpy correction (lh - L_v E)/(rho c_pd):
    # the bulk law charged Kirchhoff L_v(SST) per kg, the LES credits L_v; the
    # rho cancels here too, leaving Ch |U| dq (L_v(SST) - L_v) / c_pd.
    from legoesm.thermo import latent_heat_vaporization
    corr = (w_qv_expected
            * (float(latent_heat_vaporization(sst)) - constants.L_v) / constants.c_pd)
    assert corr < 0.0   # 292 K > T0: L_v(SST) < L_v
    w_th_expected = (ch * wind * (float(sst) - float(T_col[s])) + corr) * exner

    assert float(w_th) == pytest.approx(w_th_expected, rel=1e-12)
    assert float(w_qv) == pytest.approx(w_qv_expected, rel=1e-12)
    # Exner DIRECTION: low surface pressure AMPLIFIES the kinematic θ flux above the
    # plain temperature flux w'T' (an inverted (p_s/p_ref)^κ would damp it instead).
    assert exner > 1.0
    assert float(w_th) > ch * wind * (float(sst) - float(T_col[s]))


def test_extract_gcm_column_surface_flux_opt_in():
    """extract_gcm_column adds a prescribe='fluxes' surface BC ONLY when sst_K is given —
    default None ⇒ surface-flux-free (iter-148 behaviour byte-unchanged).  Locks the
    opt-in: no sst_K ⇒ prescribe='none' (no surface fields); with sst_K ⇒ prescribe='fluxes'
    with a positive θ/q_v flux (a warmer-than-air SST)."""
    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    # TOP-DOWN column (index 0 = top): warm surface air (last level) + a warmer SST.
    T = jnp.broadcast_to(jnp.linspace(240.0, 295.0, nlev), shape)  # noqa: N806
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 8.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    sst = jnp.full((n_lat, n_lon), 300.0)
    kw = dict(T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
              col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)))
    _, _, ls_none = extract_gcm_column(**kw)                       # default: no sst_K
    assert ls_none.prescribe == "none"
    assert ls_none.w_th_s is None and ls_none.w_qv_s is None
    _, _, ls_flux = extract_gcm_column(**kw, sst_K=sst)            # opt-in
    assert ls_flux.prescribe == "fluxes"
    assert float(ls_flux.w_th_s) > 0.0 and float(ls_flux.w_qv_s) > 0.0


def test_extract_gcm_column_uses_hybrid_pressures_over_terrain():
    """extract_gcm_column builds the GCM column's θ from the model COORDINATE's pressures
    (iter 341), not pure-sigma σ·p_s — so for a hybrid coordinate (the dycore default) over a
    terrain column (p_s != p_ref) the reference θ DIFFERS from the pure-sigma version (it was
    silently wrong before).  A pure-sigma coordinate reproduces the prior θ exactly (locked by
    test_extract_gcm_column above)."""
    from legoesm.grids.vertical import make_hybrid_levels

    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    shape = (n_lat, n_lon, nlev)
    kw = dict(
        T=jnp.full(shape, 280.0), q_v=jnp.full(shape, 5e-3),
        u=jnp.full(shape, 10.0), v=jnp.zeros(shape),
        p_s=jnp.full((n_lat, n_lon), 7.0e4),       # 700-hPa terrain column (p_s != p_ref ~1e5)
        grid=grid, col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)))
    _, theta_sigma, _ = extract_gcm_column(sigma=create_sigma_coordinate(nlev), **kw)
    _, theta_hybrid, _ = extract_gcm_column(
        sigma=make_hybrid_levels(nlev, p_top_Pa=100.0), **kw)
    # The hybrid column pressures (A·p_ref + B·p_s) differ from σ·p_s over terrain ⇒ different θ.
    assert float(jnp.max(jnp.abs(theta_hybrid - theta_sigma))) > 0.5


def test_extract_gcm_column_orographic_phis_activates_geostrophic_term():
    """iter-118 driver wiring: passing ``phis`` to extract_gcm_column threads the
    orographic surface-geopotential term (iter 117) into the geostrophic forcing —
    the returned ls_state's u_geo/v_geo differ from the flat (phis=None) case over
    terrain; ``phis=None`` (the default) is bit-identical to omitting it."""
    from legoesm import constants
    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 10.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    kw = dict(T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
              col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)))
    _, _, ls_flat = extract_gcm_column(**kw)
    lat2d = jnp.asarray(grid.lat)[:, None]
    lon2d = jnp.asarray(grid.lon)[None, :]
    phis = constants.g * (700.0 * jnp.cos(lat2d) * jnp.sin(lon2d))   # ~700 m terrain
    _, _, ls_oro = extract_gcm_column(**kw, phis=phis)
    assert ls_flat.u_geo is not None and ls_oro.u_geo is not None
    du = np.asarray(ls_oro.u_geo) - np.asarray(ls_flat.u_geo)
    dv = np.asarray(ls_oro.v_geo) - np.asarray(ls_flat.v_geo)
    assert float(np.max(np.abs(du)) + np.max(np.abs(dv))) > 1e-3   # orographic ACTIVATED
    # phis=None default is bit-identical to the flat case (unchanged production path)
    _, _, ls_none = extract_gcm_column(**kw, phis=None)
    np.testing.assert_array_equal(np.asarray(ls_none.u_geo), np.asarray(ls_flat.u_geo))


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


class _SfcEnv:
    cape_J_kg = 200.0  # noqa: N815 — matches the manifest env field name


class _SfcRec:
    grid_index = (4, 8)
    lat_deg = 20.0
    environment = _SfcEnv()


def _sfc_state(nlev=6, n_lat=8, n_lon=16):
    shape = (n_lat, n_lon, nlev)
    return dict(
        T=jnp.broadcast_to(jnp.linspace(240.0, 295.0, nlev), shape),  # warm surface (last)
        q_v=jnp.full(shape, 5e-3), u=jnp.full(shape, 8.0), v=jnp.zeros(shape),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        grid=create_latlon_grid(n_lat, n_lon, dtype=jnp.float64),
        sigma=create_sigma_coordinate(nlev))


def test_process_column_surface_flux_requires_sst():
    """config.surface_flux=True with no sst_K is a LOUD error (dispatch-hardening): a
    silently surface-flux-free LES would defeat the opt-in."""
    cfg = ColumnLESConfig(regime=_SMALL_REGIME, surface_flux=True)
    with pytest.raises(ValueError, match="surface_flux=True but no sst_K"):
        process_column(_SfcRec(), config=cfg, run_les_fn=lambda s: None, **_sfc_state())


def test_process_column_threads_sst_only_when_surface_flux(monkeypatch):
    """process_column passes sst_K to extract_gcm_column ONLY when config.surface_flux is
    set (else None ⇒ the surface-flux-free LES, iter-148 unchanged).  Spies on
    extract_gcm_column so the wiring is locked without inspecting the opaque forcing."""
    from legoesm.atmosphere.dynamics.les import column_les

    captured = {}
    real = column_les.extract_gcm_column

    def spy(**kw):
        captured["sst_K"] = kw.get("sst_K")
        return real(**kw)

    monkeypatch.setattr(column_les, "extract_gcm_column", spy)
    sst = jnp.full((8, 16), 300.0)
    fake_run = lambda s: _synthetic_plane_state(s.grid, s.height_coord)  # noqa: E731

    # OFF (default): sst_K is supplied but NOT forwarded (no surface flux).
    process_column(_SfcRec(), config=ColumnLESConfig(regime=_SMALL_REGIME),
                   run_les_fn=fake_run, sst_K=sst, **_sfc_state())
    assert captured["sst_K"] is None
    # ON: sst_K forwarded ⇒ extract_gcm_column builds the prescribe='fluxes' BC.
    process_column(_SfcRec(), config=ColumnLESConfig(regime=_SMALL_REGIME, surface_flux=True),
                   run_les_fn=fake_run, sst_K=sst, **_sfc_state())
    assert captured["sst_K"] is not None


def test_process_column_surface_flux_composes_with_multi_coefficient(monkeypatch):
    """The surface-flux opt-in composes with the MULTI-coefficient diagnosis: ONE
    surface-flux LES feeds ALL methods.  The surface flux is built into the shared
    ``setup`` (``process_column`` → ``extract_gcm_column``, line 690-695) UPSTREAM of
    ``run_column_les_pipeline``'s single/multi branch, so a surface-flux +
    simultaneous-C_K/Pr_t HPC campaign must (a) forward ``sst_K`` to
    ``extract_gcm_column`` in the MULTI path AND (b) return the ``{method: diagnosis}``
    dict.  The existing spy test covers only the SINGLE path; this guards against a
    future refactor moving the surface-flux threading into a single-only branch."""
    from legoesm.atmosphere.dynamics.les import column_les

    captured = {}
    real = column_les.extract_gcm_column

    def spy(**kw):
        captured["sst_K"] = kw.get("sst_K")
        return real(**kw)

    monkeypatch.setattr(column_les, "extract_gcm_column", spy)
    sst = jnp.full((8, 16), 300.0)
    fake_run = lambda s: _synthetic_plane_state(s.grid, s.height_coord)  # noqa: E731
    cfg = ColumnLESConfig(
        regime=_SMALL_REGIME, surface_flux=True,
        diagnosis_methods=("clubb_coefficient", "prandtl_number"),
        clubb_l_mix_max=100.0)
    out = process_column(_SfcRec(), config=cfg, run_les_fn=fake_run, sst_K=sst,
                         **_sfc_state())
    # (a) the surface flux reached the LES setup IN THE MULTI path...
    assert captured["sst_K"] is not None
    # (b) ...and the multi-coefficient dict (one LES, all methods) is returned + finite.
    assert set(out) == {"clubb_coefficient", "prandtl_number"}
    assert bool(jnp.all(jnp.isfinite(out["clubb_coefficient"].C_K)))
    assert bool(jnp.all(jnp.isfinite(out["prandtl_number"].Pr_t)))


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


def test_extract_gcm_column_surface_flux_cubed_sphere():
    """The surface-flux opt-in works on a CUBED-SPHERE state — the 3-tuple ``col_index``
    + the ``sst_K[face,i,j]`` gather (the 2nd cell-grid family, after the iter-369 MPAS
    fail-loud).  Uses a PHYSICAL state (the gather test above uses unphysical column-unique
    values): a warmer-than-air SST ⇒ ``prescribe='fluxes'`` with a positive θ/q_v flux;
    default (no sst_K) stays surface-flux-free."""
    from legoesm.grids.factory import create_grid

    res, nlev = 8, 6
    grid = create_grid("cubed_sphere", resolution=res)
    sigma = create_sigma_coordinate(nlev)
    shape = (6, res, res, nlev)
    T = jnp.broadcast_to(jnp.linspace(240.0, 295.0, nlev), shape)  # physical  # noqa: N806
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 8.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((6, res, res), 1.0e5)
    sst = jnp.full((6, res, res), 320.0)              # warmer than the surface air (≤295)
    kw = dict(T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
              col_index=(2, 3, 5), lat_rad=float(jnp.deg2rad(20.0)))
    _, _, ls_none = extract_gcm_column(**kw)
    assert ls_none.prescribe == "none"
    _, _, ls_flux = extract_gcm_column(**kw, sst_K=sst)
    assert ls_flux.prescribe == "fluxes"
    assert float(ls_flux.w_th_s) > 0.0 and float(ls_flux.w_qv_s) > 0.0


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


@pytest.mark.filterwarnings("error::FutureWarning")  # iter 211: no f64->f32 scatter
def test_process_column_real_dycore_integration():
    """MOCK-FREE end-to-end: extract -> setup -> the REAL plane-NH dycore
    (run_forced_les, a few real steps with the large-scale forcing + top
    relaxation) -> diagnose. Validates the run path the unit tests mock out
    (the compressible-Euler plane LES actually runs and stays finite)."""
    from legoesm.atmosphere.dynamics.les.column_les import (
        build_column_les_setup,
        run_forced_les,
    )
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
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


@pytest.mark.filterwarnings("error::FutureWarning")  # iter 211: no f64->f32 scatter
def test_process_column_real_dycore_surface_flux_warms_vs_noflux():
    """MOCK-FREE end-to-end for the SURFACE-FLUX opt-in (iter 364-371): the helper's
    kinematic surface flux, fed through the REAL plane-NH dycore, must (a) stay finite
    and (b) inject net heat — the surface-flux run's column-total theta' strictly
    exceeds the otherwise-identical no-flux run's.  The unit tests exercise the flux
    wiring against a MOCK LES; this is the only test that runs the helper's flux through
    the actual compressible-Euler integration, locking that prescribe='fluxes' is not a
    silent no-op and warms in the physically correct direction."""
    from legoesm.atmosphere.dynamics.les.column_les import (
        build_column_les_setup,
        run_forced_les,
    )

    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    # TOP-DOWN column (index 0 = top): warm surface air + a warmer SST ⇒ upward flux.
    T = jnp.broadcast_to(jnp.linspace(240.0, 295.0, nlev), shape)  # noqa: N806
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 8.0)
    v = jnp.zeros(shape)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    sst = jnp.full((n_lat, n_lon), 300.0)
    common = dict(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)),
    )

    def _run(ls_state):
        setup = build_column_les_setup(
            cape_J_kg=200.0, lat_rad=float(jnp.deg2rad(20.0)),
            gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls_state, config=_CONFIG,
        )
        return run_forced_les(setup, dt_s=0.5, n_steps=3)

    gcm_z, gcm_theta, ls_noflux = extract_gcm_column(**common)             # prescribe='none'
    _, _, ls_flux = extract_gcm_column(**common, sst_K=sst)                # prescribe='fluxes'
    assert ls_noflux.prescribe == "none"
    assert ls_flux.prescribe == "fluxes" and float(ls_flux.w_th_s) > 0.0

    final_noflux = _run(ls_noflux)
    final_flux = _run(ls_flux)

    # Both real integrations stay finite (no acoustic blow-up).
    for arr in (final_flux.theta_prime.data, final_flux.tracers.data,
                final_noflux.theta_prime.data):
        assert bool(jnp.all(jnp.isfinite(arr)))
    # The surface flux is NOT a silent no-op and warms in the right direction:
    # net column theta' is strictly larger with the upward surface heat flux.
    total_flux = float(jnp.sum(final_flux.theta_prime.data))
    total_noflux = float(jnp.sum(final_noflux.theta_prime.data))
    assert total_flux > total_noflux


def test_run_forced_les_rejects_acoustically_unstable_dt():
    """run_forced_les FAILS FAST on a timestep that violates the horizontal acoustic
    CFL — caught BEFORE the multi-day run, not as a mid-run blow-up (iter 103). The
    raise fires before the time loop, so this is cheap (no LES steps); the tested-
    stable dt_s=0.5 (C_a≈0.57) is covered by the real-dycore test above."""
    from legoesm.atmosphere.dynamics.les.column_les import (
        build_column_les_setup,
        extract_gcm_column,
        run_forced_les,
    )

    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    gcm_z, gcm_theta, ls = extract_gcm_column(
        T=jnp.full(shape, 280.0), q_v=jnp.full(shape, 5e-3),
        u=jnp.full(shape, 8.0), v=jnp.zeros(shape),
        p_s=jnp.full((n_lat, n_lon), 1.0e5), grid=grid, sigma=sigma,
        col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)))
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=float(jnp.deg2rad(20.0)),
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG)
    # dx=50 m, n_acoustic=6, c≈340 ⇒ dt_s=2.0 gives C_a≈2.3 > 1.0 ⇒ reject.
    with pytest.raises(ValueError, match="Courant.*exceeds the stable limit"):
        run_forced_les(setup, dt_s=2.0, n_steps=1)


def test_run_forced_les_warming_margin_rejects_marginal_dt():
    """The convective-warming margin REJECTS a dt whose RAW (rest-state) acoustic
    Courant is just BELOW 1.0 but exceeds it once inflated — a config that is marginal
    at rest but unstable once the column warms (iter 103 Codex). Computed (not a magic
    dt) so it is robust to the setup's exact c_sound."""
    from legoesm.atmosphere.dynamics.shared.cfl_diagnostic import acoustic_courant_horizontal
    from legoesm.atmosphere.dynamics.les.column_les import (
        _LES_ACOUSTIC_WARMING_MARGIN,
        build_column_les_setup,
        extract_gcm_column,
        run_forced_les,
    )

    n_lat, n_lon, nlev = 8, 16, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    gcm_z, gcm_theta, ls = extract_gcm_column(
        T=jnp.full(shape, 280.0), q_v=jnp.full(shape, 5e-3),
        u=jnp.full(shape, 8.0), v=jnp.zeros(shape),
        p_s=jnp.full((n_lat, n_lon), 1.0e5), grid=grid, sigma=sigma,
        col_index=(4, 8), lat_rad=float(jnp.deg2rad(20.0)))
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=float(jnp.deg2rad(20.0)),
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG)
    # pick dt so the RAW Courant ≈ 0.97 (∈ (1/margin, 1.0)) → margin pushes it over.
    raw_at_1 = float(acoustic_courant_horizontal(
        setup.height_coord, setup.grid, 1.0, n_acoustic_substeps=6))
    dt_marginal = 0.97 / raw_at_1
    assert 1.0 / _LES_ACOUSTIC_WARMING_MARGIN < 0.97 < 1.0   # genuinely in the band
    with pytest.raises(ValueError, match="warming margin"):
        run_forced_les(setup, dt_s=dt_marginal, n_steps=1)


def test_column_les_cli_main_wiring_monkeypatched(tmp_path, monkeypatch):
    """Drive the column-LES CLI main() with all heavy I/O stubbed: it loads the manifest
    + AMIP restart, runs process_column PER worst column (the clause-4 'spin off an LES
    for each worst column'), and SAVES the diagnosed coefficient per column. main() was
    untested (heavy I/O); a wiring regression (the per-record loop, the process_column
    call, or the np.savez key/format) would surface only at launch."""
    from types import SimpleNamespace

    import legoesm.atmosphere.dynamics.les.column_les as cl
    import legoesm.driver.restart as restart_mod
    import legoesm.grids.factory as gf
    import legoesm.grids.vertical as gv
    import legoesm.training.column_manifest as cm

    import scripts.run.run_column_les as cli

    fake_state = SimpleNamespace(
        T=jnp.zeros((1, 1, 5)), u=jnp.zeros((1, 1, 5)),
        v=jnp.zeros((1, 1, 5)), p_s=jnp.zeros((1, 1)))
    monkeypatch.setattr(gf, "create_grid", lambda gt, resolution: object())
    monkeypatch.setattr(gv, "create_sigma_coordinate", lambda nlev: object())
    monkeypatch.setattr(restart_mod, "load_restart",
                        lambda path, g, s, strict: (fake_state, jnp.zeros((1, 1, 5))))
    recs = [SimpleNamespace(grid_index=(0, 0)), SimpleNamespace(grid_index=(1, 2))]
    monkeypatch.setattr(cm, "read_manifest", lambda path: recs)

    seen = []

    def fake_process_column(rec, **kwargs):           # noqa: ARG001
        seen.append(rec.grid_index)
        return ("diag", rec.grid_index)

    monkeypatch.setattr(cl, "process_column", fake_process_column)
    monkeypatch.setattr(cl, "coefficient_value",
                        lambda diag, method: np.asarray([1.5, 2.5]))

    out = str(tmp_path / "coef.npz")
    rc = cli.main(["--manifest", "m.json", "--restart", "r.npz",
                   "--resolution", "8", "--nlev", "5", "--out", out,
                   "--method", "clubb_coefficient"])
    assert rc == 0
    assert seen == [(0, 0), (1, 2)]                   # process_column run per worst column
    loaded = np.load(out)
    assert set(loaded.files) == {"(0, 0)", "(1, 2)"}  # one saved coefficient per column
    np.testing.assert_allclose(loaded["(0, 0)"], [1.5, 2.5])


def test_realism_verdict_names_failing_criteria():
    """_realism_verdict (iter 511) turns a LESRealismBreakdown into the operator's per-column
    debug line: REALISTIC when all pass, else REJECTED naming each failing mode — so a debug
    run says whether a diagnosed coefficient can be believed and, if not, why."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import LESRealismBreakdown

    from scripts.run.run_column_les import _realism_verdict

    def _bd(turbulent=True, finite=True, thermo=True, moisture=True, rh=True):
        flags = (turbulent, finite, thermo, moisture, rh)
        return LESRealismBreakdown(
            turbulent=jnp.asarray(turbulent), finite=jnp.asarray(finite),
            thermo_consistent=jnp.asarray(thermo), moisture_physical=jnp.asarray(moisture),
            rh_ok=jnp.asarray(rh), overall=jnp.asarray(all(flags)))

    assert _realism_verdict(_bd()) == "REALISTIC"
    assert _realism_verdict(_bd(turbulent=False)) == "REJECTED(not_turbulent)"
    assert _realism_verdict(_bd(finite=False)) == "REJECTED(not_finite)"
    assert _realism_verdict(_bd(moisture=False)) == "REJECTED(moisture_runaway)"
    # multiple failures are all named, in declared order
    assert _realism_verdict(_bd(turbulent=False, thermo=False)) == \
        "REJECTED(not_turbulent,thermo_drift)"


def test_column_les_cli_reports_realism_when_les_runs(tmp_path, monkeypatch, capsys):
    """When process_column actually invokes run_les_fn (the real LES path), main() CAPTURES
    the finished state + height_coord and reports the per-column realism verdict (iter 511) —
    exercising the capture→breakdown→output wiring, not just the mocked no-LES path."""
    from types import SimpleNamespace

    import legoesm.atmosphere.dynamics.les.column_les as cl
    import legoesm.atmosphere.dynamics.les.column_les_diagnosis as cld
    import legoesm.driver.restart as restart_mod
    import legoesm.grids.factory as gf
    import legoesm.grids.vertical as gv
    import legoesm.training.column_manifest as cm

    import scripts.run.run_column_les as cli

    fake_state = SimpleNamespace(
        T=jnp.zeros((1, 1, 5)), u=jnp.zeros((1, 1, 5)),
        v=jnp.zeros((1, 1, 5)), p_s=jnp.zeros((1, 1)))
    monkeypatch.setattr(gf, "create_grid", lambda gt, resolution: object())
    monkeypatch.setattr(gv, "create_sigma_coordinate", lambda nlev: object())
    monkeypatch.setattr(restart_mod, "load_restart",
                        lambda path, g, s, strict: (fake_state, jnp.zeros((1, 1, 5))))
    monkeypatch.setattr(cm, "read_manifest",
                        lambda path: [SimpleNamespace(grid_index=(0, 0))])

    les_final = object()                                  # the "finished LES state" sentinel
    monkeypatch.setattr(cl, "run_forced_les",
                        lambda setup, *, dt_s, n_steps: les_final)

    def fake_process_column(rec, *, run_les_fn, **kwargs):  # noqa: ARG001
        run_les_fn(SimpleNamespace(height_coord="HC"))     # triggers _run → captures les_final
        return SimpleNamespace(valid=jnp.asarray([True, True, False, False]))  # 2/4 valid

    monkeypatch.setattr(cl, "process_column", fake_process_column)
    monkeypatch.setattr(cl, "coefficient_value", lambda diag, method: np.asarray([1.0]))

    seen = {}

    def fake_breakdown(state, hc):
        seen["state"], seen["hc"] = state, hc
        true = jnp.asarray(True)
        return SimpleNamespace(overall=true, turbulent=true, finite=true,
                               thermo_consistent=true, moisture_physical=true, rh_ok=true)

    monkeypatch.setattr(cld, "column_les_realism_breakdown", fake_breakdown)

    rc = cli.main(["--manifest", "m", "--restart", "r", "--resolution", "8",
                   "--nlev", "5", "--out", str(tmp_path / "c.npz"),
                   "--method", "clubb_coefficient"])
    assert rc == 0
    assert seen["state"] is les_final and seen["hc"] == "HC"   # captured state reached it
    out = capsys.readouterr().out
    assert "realism: REALISTIC" in out
    # iter 524: the per-column line ALSO reports the valid diagnosis-level count, so a
    # REALISTIC line with few/0 valid levels points at the DIAGNOSIS, not the realism gate.
    assert "2/4 valid diagnosis levels" in out


def test_run_pipeline_excludes_the_top_sponge_layer(monkeypatch):
    """relax_width_frac threads through run_column_les_pipeline (iter 513): it computes
    z_max = domain_top*(1-relax_width_frac) from the setup's height coord and AND-s out every
    diagnosis interface at/above it. Mock the diagnosis to an ALL-VALID profile spanning the
    domain so the ONLY invalidation is the sponge — pinning the z_max computation + the mask
    application precisely (below-sponge stays valid, sponge is excluded)."""
    import legoesm.atmosphere.dynamics.les.column_les as cl
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import ClubbCoefficientProfile

    gcm_z, gcm_theta, ls = _gcm_column()
    setup = build_column_les_setup(
        cape_J_kg=200.0, lat_rad=0.3,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls, config=_CONFIG)
    domain_top = float(jnp.max(jnp.asarray(setup.height_coord.z_full)))
    z_max = domain_top * 0.75                              # relax_width_frac = 0.25

    z_m = jnp.linspace(domain_top * 0.1, domain_top * 0.95, 8)
    all_valid = ClubbCoefficientProfile(
        z_m=z_m, C_K=jnp.full((8,), 0.3), valid=jnp.ones(8, dtype=bool))
    monkeypatch.setattr(cl, "diagnose_column_coefficient", lambda *a, **k: all_valid)

    out = run_column_les_pipeline(
        setup, lambda s: _synthetic_plane_state(s.grid, s.height_coord),
        method="clubb_coefficient", gate_realism=False, relax_width_frac=0.25)
    z, valid = np.asarray(out.z_m), np.asarray(out.valid)
    sponge = z >= z_max
    assert sponge.any() and (~sponge).any()               # non-vacuous: both sides present
    assert not valid[sponge].any()                        # sponge excluded
    assert valid[~sponge].all()                           # below-sponge UNCHANGED (precise z_max)


def test_valid_levels_note_distinguishes_diagnosis_rejection():
    """_valid_levels_note (iter 524) reports 'N/M valid diagnosis levels' for a per-level
    diagnosis, so a REALISTIC LES with 0 valid levels points at the DIAGNOSIS validity (not the
    realism gate, iter 507); empty for a scalar diagnosis (entrainment)."""
    from types import SimpleNamespace

    from scripts.run.run_column_les import _valid_levels_note

    prof = SimpleNamespace(valid=jnp.asarray([True, True, False, False]))
    assert _valid_levels_note(prof) == " — 2/4 valid diagnosis levels"
    none_valid = SimpleNamespace(valid=jnp.zeros(7, dtype=bool))
    assert _valid_levels_note(none_valid) == " — 0/7 valid diagnosis levels"
    # scalar diagnosis (entrainment w_e) → no per-level count
    assert _valid_levels_note(SimpleNamespace(valid=jnp.asarray(True))) == ""
    assert _valid_levels_note(SimpleNamespace()) == ""              # no .valid → empty
