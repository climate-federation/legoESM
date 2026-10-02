"""MPAS land surface boundary knobs (speckle fix, 2026-07-23).

Covers the full chain:
  - ``land_lapse_adjusted_surface_temperature`` (forcing.surface_utils):
    lapse math, ocean identity, below-sea-level clip.
  - ``beta_limited_surface_humidity`` (turbulence.surface_layer): swamp
    identity at beta=1, closed surface at beta=0 over full land, ocean
    fraction untouched.
  - Factory guards: ``make_turbulence_physics`` / ``make_physics`` REFUSE
    the knobs for non-MPAS model types (no silently-inert configuration).
  - ``_make_mpas_turbulence`` end-to-end on a small SCVT mesh: beta < 1
    reduces the land-cell surface moistening, leaves ocean cells
    bit-identical.
  - ``ExperimentConfig.validate_strict``: pipeline land-tile flags are
    refused on the MPAS lane (they never execute there); the MPAS knobs are
    refused on non-MPAS lanes; bounds are enforced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.forcing.surface_utils import (
    land_lapse_adjusted_surface_temperature,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    beta_limited_surface_humidity,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.integration import (
    _make_mpas_turbulence,
    make_turbulence_physics,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_lapse_ocean_fraction_unchanged():
    T = jnp.array([300.0, 290.0])
    f_land = jnp.array([0.0, 0.0])
    z = jnp.array([0.0, 3000.0])
    out = land_lapse_adjusted_surface_temperature(T, f_land, z, 6.5e-3)
    np.testing.assert_allclose(np.asarray(out), np.asarray(T))


def test_lapse_land_cooling_matches_rate():
    # full land at 1 km with 6.5 K/km -> exactly -6.5 K
    out = land_lapse_adjusted_surface_temperature(
        jnp.array([300.0]), jnp.array([1.0]), jnp.array([1000.0]), 6.5e-3)
    np.testing.assert_allclose(np.asarray(out), [300.0 - 6.5], rtol=1e-12)
    # fractional land scales linearly
    out_half = land_lapse_adjusted_surface_temperature(
        jnp.array([300.0]), jnp.array([0.5]), jnp.array([1000.0]), 6.5e-3)
    np.testing.assert_allclose(np.asarray(out_half), [300.0 - 3.25], rtol=1e-12)


def test_lapse_below_sea_level_clipped_not_warmed():
    out = land_lapse_adjusted_surface_temperature(
        jnp.array([300.0]), jnp.array([1.0]), jnp.array([-400.0]), 6.5e-3)
    np.testing.assert_allclose(np.asarray(out), [300.0])


def test_lapse_zero_rate_identity():
    T = jnp.array([288.0, 301.5])
    out = land_lapse_adjusted_surface_temperature(
        T, jnp.array([1.0, 0.3]), jnp.array([500.0, 1500.0]), 0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(T))


def test_beta_one_is_saturated_identity():
    q_sat = jnp.array([0.025, 0.020])
    q_air = jnp.array([0.010, 0.015])
    f_land = jnp.array([1.0, 0.4])
    out = beta_limited_surface_humidity(q_sat, q_air, f_land, 1.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(q_sat))


def test_beta_zero_full_land_closes_surface():
    q_sat = jnp.array([0.025])
    q_air = jnp.array([0.010])
    out = beta_limited_surface_humidity(q_sat, q_air, jnp.array([1.0]), 0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(q_air))


def test_beta_ocean_fraction_stays_saturated():
    q_sat = jnp.array([0.025])
    q_air = jnp.array([0.010])
    out = beta_limited_surface_humidity(q_sat, q_air, jnp.array([0.0]), 0.3)
    np.testing.assert_allclose(np.asarray(out), np.asarray(q_sat))


def test_beta_half_land_half_beta_interpolates():
    # q_sfc = q_air + (1 - f*(1-beta)) * (q_sat - q_air)
    q_sat, q_air, f, beta = 0.030, 0.010, 0.5, 0.6
    out = beta_limited_surface_humidity(
        jnp.array([q_sat]), jnp.array([q_air]), jnp.array([f]), beta)
    expected = q_air + (1.0 - f * (1.0 - beta)) * (q_sat - q_air)
    np.testing.assert_allclose(np.asarray(out), [expected], rtol=1e-12)


# ---------------------------------------------------------------------------
# Factory guards: refuse silently-inert configurations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["hydrostatic", "nonhydrostatic",
                                        "spectral_pe"])
def test_make_turbulence_physics_refuses_knobs_off_mpas(model_type):
    with pytest.raises(ValueError, match="MPAS"):
        make_turbulence_physics(
            TurbulenceConfig(scheme="louis"), model_type, 300.0,
            f_land=jnp.zeros(4))
    with pytest.raises(ValueError, match="MPAS"):
        make_turbulence_physics(
            TurbulenceConfig(scheme="louis"), model_type, 300.0,
            land_beta=0.5)


def test_make_physics_refuses_knobs_off_mpas():
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    cfg = PhysicsConfig()
    with pytest.raises(ValueError, match="MPAS-only"):
        make_physics(cfg, model_type="hydrostatic", dt=300.0,
                     land_beta=0.5)


# ---------------------------------------------------------------------------
# End-to-end on a small SCVT mesh: beta throttles land moistening only
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mpas_mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)


@pytest.fixture(scope="module")
def sigma_coord():
    from legoesm.grids.vertical import create_sigma_coordinate
    return create_sigma_coordinate(8, sigma_top=0.05)


@pytest.fixture(scope="module")
def mpas_state(mpas_mesh, sigma_coord):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    state = held_suarez_init_mpas(
        mpas_mesh, sigma_coord, T_init=290.0, perturbation_amplitude=0.0)
    ncell = state.T.data.shape[0]
    nlev = state.T.data.shape[1]
    # Moist but subsaturated boundary layer so a warm anchor evaporates.
    q_v = jnp.full((ncell, nlev), 0.008, dtype=state.T.data.dtype)
    return state._replace(tracers={"q_v": q_v})


def test_mpas_turbulence_beta_throttles_land_only(mpas_mesh, sigma_coord,
                                                  mpas_state):
    ncell = mpas_state.T.data.shape[0]
    # Land on one hemisphere of cells, ocean on the other.
    f_land = jnp.asarray(
        (np.arange(ncell) < ncell // 2).astype(np.float64))
    T_sfc = jnp.full((ncell,), 302.0)
    forcing = {"T_sfc": T_sfc}
    tc = TurbulenceConfig(scheme="louis")

    fn_swamp = _make_mpas_turbulence(tc, 300.0)
    fn_beta = _make_mpas_turbulence(tc, 300.0, f_land=f_land, land_beta=0.4)

    tend_swamp = fn_swamp(mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
    tend_beta = fn_beta(mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
    if isinstance(tend_swamp, tuple):
        tend_swamp = tend_swamp[0]
    if isinstance(tend_beta, tuple):
        tend_beta = tend_beta[0]

    dq_swamp = np.asarray(tend_swamp.tracer_tendencies["q_v"])[..., -1]
    dq_beta = np.asarray(tend_beta.tracer_tendencies["q_v"])[..., -1]
    land = np.asarray(f_land) > 0.5
    ocean = ~land

    # Ocean cells: bit-identical (the knob must not touch them).
    np.testing.assert_array_equal(dq_beta[ocean], dq_swamp[ocean])
    # Land cells: swamp moistens the surface layer; beta=0.4 moistens LESS
    # (strictly, since the anchor is 12 K warmer than the air => q_sat >
    # q_air everywhere).
    assert (dq_swamp[land] > 0.0).all(), "warm swamp must moisten land BL"
    assert (dq_beta[land] < dq_swamp[land]).all()
    # And the throttled flux is still non-negative (beta in (0,1) cannot
    # reverse the gradient sign).
    assert (dq_beta[land] >= 0.0).all()


def _land_forcing(ncell, shflx=None, lhflx=None):
    f = {"T_sfc": jnp.full((ncell,), 300.0)}
    if shflx is not None:
        f["shflx_land"] = jnp.full((ncell,), shflx)
        f["lhflx_land"] = jnp.full((ncell,), lhflx)
    return f


# Which schemes may be handed the land model's own turbulent fluxes is decided
# by one property: does the kernel declare a ``surface_flux`` argument?  It is
# NOT the same question as whether the scheme carries prognostic turbulent
# energy — clubb and clubb_lite do both.
_LAND_FLUX_REFUSED = ["smagorinsky", "holtslag_boville", "ysu", "tke",
                      "mynn25", "edmf"]
_LAND_FLUX_ACCEPTED = ["louis", "clubb", "clubb_lite"]


def test_the_admissible_set_matches_the_hand_written_split():
    """Ties the signature scan to the two lists the tests below are built from.

    Written out by hand on purpose: if a kernel gains or loses the argument,
    this fails and forces the parametrised cases to move with it, instead of
    both sides drifting together and proving nothing.
    """
    from legoesm.atmosphere.physics.turbulence.integration import (
        schemes_accepting_surface_flux,
    )
    assert set(schemes_accepting_surface_flux()) == set(_LAND_FLUX_ACCEPTED)
    assert not set(schemes_accepting_surface_flux()) & set(_LAND_FLUX_REFUSED)


@pytest.mark.parametrize("scheme", _LAND_FLUX_REFUSED)
def test_land_fluxes_are_refused_by_a_scheme_that_cannot_consume_them(
        mpas_mesh, sigma_coord, mpas_state, scheme):
    """Handing the land model's fluxes to a scheme that computes its own.

    These kernels take no ``surface_flux`` argument.  Passing the fluxes
    anyway used to raise TypeError from inside a traced column; silently
    dropping them would be worse still, because the run would advertise land
    coupling while the atmosphere applied a surface flux the land model never
    solved.
    """
    ncell = mpas_state.T.data.shape[0]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme=scheme), 300.0, f_land=jnp.ones((ncell,)))
    with pytest.raises(ValueError, match="cannot be handed the land surface"):
        fn(mpas_state, mpas_mesh, sigma_coord,
           forcing=_land_forcing(ncell, 40.0, 90.0))


@pytest.mark.parametrize("scheme", _LAND_FLUX_ACCEPTED)
def test_land_fluxes_actually_reach_every_scheme_the_guard_admits(
        mpas_mesh, sigma_coord, mpas_state, scheme):
    """The other half of the guard, for EVERY scheme it lets through.

    A guard that only ever refuses would be satisfied by a coupling that never
    works, and declaring the keyword is not the same as using it.  A large
    sensible-heat flux into the lowest layer must WARM it, so the direction is
    pinned too, not just that something moved: a sign-inverted hand-over would
    otherwise pass.
    """
    ncell = mpas_state.T.data.shape[0]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme=scheme), 300.0, f_land=jnp.ones((ncell,)))

    without = fn(mpas_state, mpas_mesh, sigma_coord,
                 forcing=_land_forcing(ncell))
    # Both far larger than anything the bulk formula produces here, so neither
    # the difference nor its sign can be round-off — and DELIBERATELY unequal,
    # so a consumer that swapped the sensible and latent members of the flux
    # tuple could not satisfy both assertions below.
    with_flux = fn(mpas_state, mpas_mesh, sigma_coord,
                   forcing=_land_forcing(ncell, 400.0, 40.0))
    if isinstance(without, tuple):
        without, with_flux = without[0], with_flux[0]
    # ``dT_dt`` is a Field on this lane; ``.data`` is (nCells, nlev).
    dT_without = np.asarray(without.dT_dt.data)[:, -1]
    dT_with = np.asarray(with_flux.dT_dt.data)[:, -1]
    assert not np.allclose(dT_without, dT_with), (
        f"{scheme}: the land sensible-heat flux was accepted and then ignored")
    assert (dT_with > dT_without).all(), (
        f"{scheme}: +400 W/m2 into the surface layer must warm it; the "
        "hand-over has the wrong sign")

    # The latent half is the headline of this coupling and needs its own pin:
    # a kernel that consumes the sensible flux and drops the latent one would
    # pass every assertion above.
    wetter = fn(mpas_state, mpas_mesh, sigma_coord,
                forcing=_land_forcing(ncell, 400.0, 400.0))
    if isinstance(wetter, tuple):
        wetter = wetter[0]
    dq_without = np.asarray(without.tracer_tendencies["q_v"])[:, -1]
    dq_with = np.asarray(with_flux.tracer_tendencies["q_v"])[:, -1]
    dq_wetter = np.asarray(wetter.tracer_tendencies["q_v"])[:, -1]
    assert not np.allclose(dq_without, dq_with), (
        f"{scheme}: the land latent-heat flux was accepted and then ignored")
    assert (dq_wetter > dq_with).all(), (
        f"{scheme}: raising the land latent flux from 40 to 400 W/m2 must "
        "moisten the surface layer more; the hand-over has the wrong sign")


def test_mpas_turbulence_beta_one_bit_identical(mpas_mesh, sigma_coord,
                                                 mpas_state):
    """beta=1 with f_land supplied must be byte-identical to no knobs."""
    ncell = mpas_state.T.data.shape[0]
    f_land = jnp.asarray((np.arange(ncell) % 2).astype(np.float64))
    forcing = {"T_sfc": jnp.full((ncell,), 300.0)}
    tc = TurbulenceConfig(scheme="louis")
    base = _make_mpas_turbulence(tc, 300.0)(
        mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
    with_knob = _make_mpas_turbulence(tc, 300.0, f_land=f_land, land_beta=1.0)(
        mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
    if isinstance(base, tuple):
        base, with_knob = base[0], with_knob[0]
    np.testing.assert_array_equal(
        np.asarray(base.tracer_tendencies["q_v"]),
        np.asarray(with_knob.tracer_tendencies["q_v"]))


# ---------------------------------------------------------------------------
# validate_strict lane guards
# ---------------------------------------------------------------------------

def _mpas_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    # Turbulence + an elevation-file source by default so the land-boundary
    # knobs are non-inert (the inert corners are tested explicitly below).
    kw.setdefault("turbulence", "louis")
    kw.setdefault("topography", "elevation.nc")
    kw.setdefault("radiation", "gray")
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        **kw,
    )


def _cdgrid_cfg(**kw):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


@pytest.mark.parametrize("knob", ["mpas_land_lapse_K_per_km", "mpas_land_beta"])
def test_validate_mpas_accepts_land_boundary_knobs(knob):
    cfg = _mpas_cfg(**{knob: 6.5 if knob == "mpas_land_lapse_K_per_km" else 0.6})
    cfg.validate_strict()
    for topography in ("flat", "gaussian"):
        idealized = cfg._replace(topography=topography)
        with pytest.raises(ValueError, match=f"topography={topography!r}.*all-zero"):
            idealized.validate_strict()
        idealized._replace(land_mask_path="land_mask.nc").validate_strict()


@pytest.mark.parametrize("flag", ["slab_land_active", "land_soil_bucket",
                                  "surface_tiled"])
def test_validate_mpas_refuses_pipeline_land_flags(flag):
    kw = {flag: True}
    if flag == "land_soil_bucket":
        kw["slab_land_active"] = True   # bucket alone already fails elsewhere
    if flag == "surface_tiled":
        kw["slab_land_active"] = True   # tiled alone already fails elsewhere
        kw["turbulence"] = "louis"
    with pytest.raises(ValueError, match="silently inert on the MPAS lane"):
        _mpas_cfg(**kw).validate_strict()


def test_validate_cdgrid_refuses_mpas_knobs():
    with pytest.raises(ValueError, match="MPAS-lane"):
        _cdgrid_cfg(mpas_land_beta=0.6).validate_strict()
    with pytest.raises(ValueError, match="MPAS-lane"):
        _cdgrid_cfg(mpas_land_lapse_K_per_km=6.5).validate_strict()


def test_validate_refuses_inert_corners():
    """Codex F1-F3: a knob whose machinery is off must be refused, not
    silently accepted."""
    with pytest.raises(ValueError, match="silently inert"):
        _mpas_cfg(mpas_land_lapse_K_per_km=6.5,
                  radiation="none").validate_strict()
    with pytest.raises(ValueError, match="silently inert"):
        _mpas_cfg(mpas_land_beta=0.6, turbulence="none").validate_strict()
    for topography in ("flat", "gaussian"):
        with pytest.raises(ValueError, match=f"topography={topography!r}.*all-zero"):
            _mpas_cfg(mpas_land_beta=0.6, topography=topography).validate_strict()
    # beta without radiation is fine (the turbulence anchor falls back but
    # the humidity throttle still applies).
    _mpas_cfg(mpas_land_beta=0.6, radiation="none").validate_strict()


def test_validate_lane_guard_keys_on_grid_type_too():
    """Codex F4: _run_mpas dispatch keys on grid_type; a grid_type='mpas'
    config with the default discretization must still be treated as the
    MPAS lane by the guard (pipeline land flags refused with the
    MPAS-lane message, not the wrong-lane one)."""
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0),   # default discretization
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        turbulence="louis", topography="gaussian",
        slab_land_active=True,
    )
    with pytest.raises(ValueError, match="silently inert on the MPAS lane"):
        cfg.validate_strict()


def test_validate_bounds():
    with pytest.raises(ValueError, match="mpas_land_beta"):
        _mpas_cfg(mpas_land_beta=1.5).validate_strict()
    with pytest.raises(ValueError, match="mpas_land_beta"):
        _mpas_cfg(mpas_land_beta=float("nan")).validate_strict()
    with pytest.raises(ValueError, match="mpas_land_lapse_K_per_km"):
        _mpas_cfg(mpas_land_lapse_K_per_km=-1.0).validate_strict()
    with pytest.raises(ValueError, match="mpas_land_lapse_K_per_km"):
        _mpas_cfg(mpas_land_lapse_K_per_km=25.0).validate_strict()


# ---------------------------------------------------------------------------
# Land-model surface stress over the land fraction (mpas_land_stress_from_land)
# ---------------------------------------------------------------------------

def test_blend_land_stress_pure_helper():
    """Bulk convention (stress opposes the wind), land fraction only, held
    columns keep the bulk stress."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        blend_land_surface_stress,
    )
    u = jnp.array([3.0, 3.0, 3.0, 3.0, 0.0])
    v = jnp.array([4.0, 4.0, 4.0, 4.0, 0.0])          # |V| = 5
    tx = jnp.array([-0.03, -0.03, -0.03, -0.03, 0.0])  # bulk |tau| = 0.05 Pa
    ty = jnp.array([-0.04, -0.04, -0.04, -0.04, 0.0])
    mag = jnp.array([0.05, 0.5, 0.0, 0.5, 0.5])
    f = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0])
    bx, by = blend_land_surface_stress(tx, ty, mag, u, v, f)
    bx, by = np.asarray(bx), np.asarray(by)
    # same magnitude as the bulk -> the bulk vector (direction and sign)
    np.testing.assert_allclose([bx[0], by[0]], [-0.03, -0.04], rtol=1e-12)
    # 10x the magnitude, still opposing the wind
    np.testing.assert_allclose([bx[1], by[1]], [-0.3, -0.4], rtol=1e-12)
    # held (zero land magnitude) -> bulk
    np.testing.assert_allclose([bx[2], by[2]], [-0.03, -0.04], rtol=1e-12)
    # ocean column untouched
    np.testing.assert_array_equal([bx[3], by[3]], [-0.03, -0.04])
    # calm air -> no direction, no stress (finite)
    np.testing.assert_array_equal([bx[4], by[4]], [0.0, 0.0])
    # linear in the land fraction
    hx, _ = blend_land_surface_stress(tx[:2], ty[:2], mag[:2], u[:2], v[:2],
                                      jnp.array([0.5, 0.5]))
    np.testing.assert_allclose(np.asarray(hx)[1], 0.5 * (-0.03) + 0.5 * (-0.3),
                               rtol=1e-12)


def test_blend_land_stress_gradient_finite_in_calm_air_and_held():
    """Calm air and a held (zero) land stress must not give NaN gradients."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        land_stress_into_surface_flux,
    )

    def f(u, v, mag):
        z = jnp.zeros(1)
        out = land_stress_into_surface_flux(
            (jnp.array([-0.01]), z, z, z, jnp.array([0.09])), mag, u, v,
            jnp.array([1.0]), jnp.array([1.2]))
        return jnp.sum(out[0] + out[1] + out[4])

    for u, mag in ((0.0, 0.3), (5.0, 0.0), (0.0, 0.0)):
        g = jax.grad(f, argnums=(0, 1, 2))(
            jnp.array([u]), jnp.array([0.0]), jnp.array([mag]))
        assert all(np.isfinite(np.asarray(x)).all() for x in g), (u, mag, g)


def test_land_stress_ustar_bulk_off_land_rebuilt_on_land():
    """ustar: bulk bit-for-bit on ocean-only and held columns; sqrt(|tau|/rho)
    of the blended stress where the land stress is used."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        land_stress_into_surface_flux,
    )
    n = 4
    tx = jnp.full(n, -0.03); ty = jnp.full(n, -0.04)       # |tau| 0.05 Pa
    sh = jnp.full(n, 10.0); lh = jnp.full(n, 20.0)
    us_bulk = jnp.full(n, 0.123)        # deliberately != sqrt(0.05/1.25)
    rho = jnp.full(n, 1.25)
    mag = jnp.array([0.5, 0.0, 0.5, 0.5])          # column 1 held
    f = jnp.array([1.0, 1.0, 0.0, 0.5])            # column 2 ocean-only
    out = land_stress_into_surface_flux(
        (tx, ty, sh, lh, us_bulk), mag, jnp.full(n, 3.0), jnp.full(n, 4.0),
        f, rho)
    us = np.asarray(out[4])
    assert us[1] == 0.123 and us[2] == 0.123
    np.testing.assert_allclose(us[0], np.sqrt(0.5 / 1.25), rtol=1e-9)
    np.testing.assert_allclose(us[3], np.sqrt(0.5 * (0.05 + 0.5) / 1.25),
                               rtol=1e-9)
    np.testing.assert_allclose([float(out[0][3]), float(out[1][3])],
                               [-0.165, -0.22], rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(out[2]), np.asarray(sh))
    np.testing.assert_array_equal(np.asarray(out[3]), np.asarray(lh))


def _windy(mpas_state):
    rng = np.random.default_rng(0)
    u_e = jnp.asarray(10.0 * rng.standard_normal(mpas_state.u.data.shape))
    return mpas_state._replace(u=mpas_state.u.replace(data=u_e))


def _du_low(out):
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out.du_dt.data)[:, -1]


@pytest.mark.parametrize("scheme", ["louis", "clubb"])
def test_land_stress_reaches_the_momentum_tendency(mpas_mesh, sigma_coord,
                                                   mpas_state, scheme):
    """A land stress far above the bulk one must DECELERATE the lowest-level
    wind more (sign pinned via the kinetic-energy tendency); a zero (held)
    land stress and an all-ocean column must reproduce the bulk stress."""
    st = _windy(mpas_state)
    ncell = st.T.data.shape[0]
    u_low = np.asarray(st.u.data)[:, -1]
    land = _make_mpas_turbulence(
        TurbulenceConfig(scheme=scheme), 300.0, f_land=jnp.ones((ncell,)))
    base_f = _land_forcing(ncell, 0.0, 0.0)
    bulk = _du_low(land(st, mpas_mesh, sigma_coord, forcing=base_f))
    strong = _du_low(land(st, mpas_mesh, sigma_coord, forcing=dict(
        base_f, taumag_land=jnp.full((ncell,), 5.0))))
    held = _du_low(land(st, mpas_mesh, sigma_coord, forcing=dict(
        base_f, taumag_land=jnp.zeros((ncell,)))))
    assert np.sum(u_low * strong) < np.sum(u_low * bulk) < 0.0, (
        f"{scheme}: a 5 Pa land stress must remove lowest-level kinetic "
        "energy faster than the bulk stress")
    np.testing.assert_allclose(held, bulk, rtol=1e-10, atol=1e-14)

    ocean = _make_mpas_turbulence(
        TurbulenceConfig(scheme=scheme), 300.0, f_land=jnp.zeros((ncell,)))
    o_bulk = _du_low(ocean(st, mpas_mesh, sigma_coord, forcing=base_f))
    o_with = _du_low(ocean(st, mpas_mesh, sigma_coord, forcing=dict(
        base_f, taumag_land=jnp.full((ncell,), 5.0))))
    np.testing.assert_allclose(o_with, o_bulk, rtol=1e-10, atol=1e-14)


def test_land_stress_without_land_heat_fluxes_is_refused(
        mpas_mesh, sigma_coord, mpas_state):
    ncell = mpas_state.T.data.shape[0]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme="louis"), 300.0, f_land=jnp.ones((ncell,)))
    with pytest.raises(ValueError, match="taumag_land"):
        fn(mpas_state, mpas_mesh, sigma_coord, forcing=dict(
            _land_forcing(ncell), taumag_land=jnp.ones((ncell,))))


def test_validate_land_stress_flag_needs_the_land_flux_handoff():
    with pytest.raises(ValueError, match="mpas_land_stress_from_land.*requires"):
        _mpas_cfg(mpas_land_stress_from_land=True).validate_strict()
    with pytest.raises(ValueError, match="mpas_land_stress_from_land is an MPAS"):
        _cdgrid_cfg(mpas_land_stress_from_land=True).validate_strict()
