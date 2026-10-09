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

from legoesm import constants
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
    schemes_accepting_surface_flux,
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
        # The land's own water flux must accompany its latent heat (pair rule).
        f["evap_land"] = jnp.full((ncell,), lhflx / constants.L_v)
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


def test_land_latent_heat_without_its_water_is_refused(mpas_mesh, sigma_coord,
                                                       mpas_state):
    """The pair rule: a land latent heat flux must arrive with the land's own
    water flux.  Inverting it with the ocean's L_v(T_sfc) would lose the
    snow-sublimation and canopy shares of the water."""
    ncell = mpas_state.T.data.shape[0]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme="louis"), 300.0, f_land=jnp.ones((ncell,)))
    f = _land_forcing(ncell, 40.0, 90.0)
    del f["evap_land"]
    with pytest.raises(ValueError, match="must come with"):
        fn(mpas_state, mpas_mesh, sigma_coord, forcing=f)


def test_undefined_land_fluxes_over_ocean_cells_do_not_poison(mpas_mesh,
                                                              sigma_coord,
                                                              mpas_state):
    """Land values over pure-ocean cells may be NaN (the land model never solved
    them).  ``(1 - f) * ocean + f * NaN`` is NaN even at f = 0, so every land
    member of the blend (heat, latent heat AND water) is masked before it is
    weighted.  Ocean cells must be bit-identical to the no-land run."""
    ncell = mpas_state.T.data.shape[0]
    f_land = jnp.asarray((np.arange(ncell) < ncell // 2).astype(np.float64))
    fn_land = _make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0,
                                    f_land=f_land)
    fn_none = _make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0)
    land = np.asarray(f_land) > 0.5
    nan_over_ocean = jnp.where(f_land > 0.0, 1.0, jnp.nan)
    f = {"T_sfc": jnp.full((ncell,), 300.0),
         "shflx_land": 40.0 * nan_over_ocean,
         "lhflx_land": 90.0 * nan_over_ocean,
         "evap_land": 90.0 / constants.L_v * nan_over_ocean}
    with_land = fn_land(mpas_state, mpas_mesh, sigma_coord, forcing=f)
    without = fn_none(mpas_state, mpas_mesh, sigma_coord,
                      forcing={"T_sfc": f["T_sfc"]})
    if isinstance(with_land, tuple):
        with_land, without = with_land[0], without[0]
    for name, a, b in (("dT_dt", with_land.dT_dt.data, without.dT_dt.data),
                       ("q_v", with_land.tracer_tendencies["q_v"],
                        without.tracer_tendencies["q_v"])):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all(), f"{name}: NaN leaked through the land blend"
        np.testing.assert_array_equal(a[~land], b[~land], err_msg=name)


# ---------------------------------------------------------------------------
# Land-model surface stress over the land fraction (mpas_land_stress_from_land)
# ---------------------------------------------------------------------------

from legoesm import constants as _const


def _lsf(n, **kw):
    """Inputs of land_stress_into_surface_flux for n columns; wind (3, 4)."""
    d = dict(
        surface_flux=(jnp.full(n, -0.03), jnp.full(n, -0.04), jnp.full(n, 10.0),
                      jnp.full(n, 20.0), jnp.full(n, 0.123)),
        tau_land_mag=jnp.full(n, 0.5), land_valid=jnp.ones(n, bool),
        z0m=jnp.full(n, 1.0), d=jnp.full(n, 10.0), u=jnp.full(n, 3.0),
        v=jnp.full(n, 4.0), z_low=jnp.full(n, 50.0), f_land=jnp.ones(n),
        rho_sfc=jnp.full(n, 1.25))
    d.update(kw)
    return d


def _call(**kw):
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        land_stress_into_surface_flux,
    )
    return land_stress_into_surface_flux(**kw)


def test_land_stress_direction_sign_and_fraction_isolation():
    """Opposes the wind (bulk convention), any direction (45 degrees too);
    blends linearly by land fraction (25/50/100 %), ocean-only bit-identical;
    heat fluxes untouched."""
    n = 4
    f = jnp.array([1.0, 0.5, 0.25, 0.0])
    out = _call(**_lsf(n, f_land=f))
    tx, ty = np.asarray(out[0]), np.asarray(out[1])
    np.testing.assert_allclose([tx[0], ty[0]], [-0.3, -0.4], rtol=1e-9)
    for k, fk in ((1, 0.5), (2, 0.25)):
        np.testing.assert_allclose(
            [tx[k], ty[k]], [(1 - fk) * -0.03 + fk * -0.3,
                             (1 - fk) * -0.04 + fk * -0.4], rtol=1e-12)
    np.testing.assert_array_equal([tx[3], ty[3]], [-0.03, -0.04])
    np.testing.assert_array_equal(np.asarray(out[2]), 10.0)
    np.testing.assert_array_equal(np.asarray(out[3]), 20.0)
    # 45-degree wind: magnitude kept, components equal and negative
    o45 = _call(**_lsf(1, u=jnp.array([5.0]), v=jnp.array([5.0])))
    np.testing.assert_allclose([float(o45[0][0]), float(o45[1][0])],
                               [-0.5 / np.sqrt(2)] * 2, rtol=1e-9)


def test_land_stress_magnitude_follows_land_not_wind_between_land_steps():
    """Between land steps the solved magnitude is held and only the direction
    follows the current wind (documented cadence)."""
    for u, v in ((3.0, 4.0), (-8.0, 1.0), (0.5, -0.2)):
        o = _call(**_lsf(1, u=jnp.array([u]), v=jnp.array([v])))
        np.testing.assert_allclose(np.hypot(float(o[0][0]), float(o[1][0])),
                                   0.5, rtol=1e-9)
        assert float(o[0][0]) * u + float(o[1][0]) * v < 0.0


def test_land_stress_before_any_valid_solve_is_neutral_land_drag_not_ocean():
    """No valid land solve yet (start / restart) or a non-finite value: the
    neutral log-law drag of the column's static land roughness, opposing the
    wind; never the bulk (ocean) stress."""
    rho, z, d, z0, U2 = 1.25, 50.0, 10.0, 1.0, 25.0
    expect = rho * (_const.kappa_vk / np.log((z - d) / z0)) ** 2 * U2
    for kw in (dict(land_valid=jnp.zeros(1, bool)),
               dict(tau_land_mag=jnp.array([jnp.nan])),
               dict(tau_land_mag=jnp.array([jnp.inf]))):
        o = _call(**_lsf(1, **kw))
        np.testing.assert_allclose([float(o[0][0]), float(o[1][0])],
                                   [-expect * 0.6, -expect * 0.8], rtol=1e-9)
    # displacement at/above the lowest level: log argument clamped (Cd <= 0.04)
    o = _call(**_lsf(1, land_valid=jnp.zeros(1, bool), d=jnp.array([60.0])))
    np.testing.assert_allclose(np.hypot(float(o[0][0]), float(o[1][0])),
                               rho * (_const.kappa_vk / 2.0) ** 2 * U2,
                               rtol=1e-9)


def test_land_stress_ignores_fill_values_on_columns_without_land():
    """NaN roughness / stress on a column with no land (fill values) must leave
    it exactly bulk, with finite gradients."""
    nan = jnp.array([jnp.nan])
    kw = dict(f_land=jnp.zeros(1), z0m=nan, d=nan, tau_land_mag=nan)
    for valid in (True, False):
        o = _call(**_lsf(1, land_valid=jnp.array([valid]), **kw))
        np.testing.assert_array_equal([float(x[0]) for x in o],
                                      [-0.03, -0.04, 10.0, 20.0, 0.123])
        g = jax.grad(lambda u: jnp.sum(jnp.stack(_call(**_lsf(
            1, u=u, land_valid=jnp.array([valid]), **kw)))))(jnp.array([3.0]))
        assert np.isfinite(np.asarray(g)).all()


def test_hold_last_valid_land_stress():
    """Held or non-finite solve keeps the previous magnitude; validity latches."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        hold_last_valid_land_stress,
    )
    prev = jnp.array([0.1, 0.2, 0.3, 0.4])
    pv = jnp.array([True, False, True, False])
    new = jnp.array([0.9, 0.8, jnp.nan, jnp.inf])
    held = jnp.array([0.0, 1.0, 0.0, 0.0])
    mag, valid, fresh = hold_last_valid_land_stress(prev, pv, new, held)
    np.testing.assert_array_equal(mag, [0.9, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(valid, [True, False, True, False])
    np.testing.assert_array_equal(fresh, [True, False, False, False])
    mag, valid, fresh = hold_last_valid_land_stress(
        mag, valid, jnp.full(4, 0.5), jnp.zeros(4))
    np.testing.assert_array_equal(mag, [0.5] * 4)
    assert bool(valid.all()) and bool(fresh.all())


def test_land_stress_ustar_bulk_off_land_rebuilt_on_land():
    """ustar: bulk bit-for-bit on ocean-only columns, sqrt(|tau|/rho) of the
    blended stress wherever there is land."""
    f = jnp.array([1.0, 0.5, 0.0])
    o = _call(**_lsf(3, f_land=f))
    us = np.asarray(o[4])
    assert us[2] == 0.123
    np.testing.assert_allclose(us[0], np.sqrt(0.5 / 1.25), rtol=1e-9)
    np.testing.assert_allclose(us[1], np.sqrt(0.5 * (0.05 + 0.5) / 1.25),
                               rtol=1e-9)


def test_land_stress_gradient_finite_calm_seed_and_nan():
    """Calm air, the neutral seed, a clamped displacement and a NaN land value
    must not produce NaN gradients."""
    def f(u, mag, valid, dd):
        o = _call(**_lsf(1, u=u, v=jnp.zeros(1), tau_land_mag=mag,
                         land_valid=valid, d=dd))
        return jnp.sum(o[0] + o[1] + o[4])
    for u, mag, valid, dd in ((0.0, 0.3, True, 10.0), (5.0, 0.3, False, 10.0),
                              (0.0, 0.0, False, 10.0), (5.0, np.nan, True, 10.0),
                              (5.0, 0.3, False, 60.0)):
        g = jax.grad(f, argnums=(0, 1, 3))(
            jnp.array([u]), jnp.array([mag]), jnp.array([valid]),
            jnp.array([dd]))
        assert all(np.isfinite(np.asarray(x)).all() for x in g), (u, mag, g)


def _windy(mpas_state):
    rng = np.random.default_rng(0)
    u_e = jnp.asarray(10.0 * rng.standard_normal(mpas_state.u.data.shape))
    return mpas_state._replace(u=mpas_state.u.replace(data=u_e))


def _du_low(out):
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out.du_dt.data)[:, -1]


def _stress_forcing(ncell, mag, valid=True):
    return dict(_land_forcing(ncell, 0.0, 0.0),
                taumag_land=jnp.full((ncell,), mag),
                taumag_land_valid=jnp.full((ncell,), valid),
                z0m_land=jnp.full((ncell,), 0.5), d_land=jnp.full((ncell,), 5.0))


@pytest.mark.parametrize("scheme", schemes_accepting_surface_flux())
def test_land_stress_reaches_the_momentum_tendency(mpas_mesh, sigma_coord,
                                                   mpas_state, scheme):
    """Every scheme the land stress can be handed to (no scheme may recompute
    the stress from the bulk law and drop it): a land stress far above the
    bulk one must DECELERATE the lowest-level
    wind more (sign pinned via the kinetic-energy tendency); an all-ocean
    column must reproduce the bulk stress."""
    st = _windy(mpas_state)
    ncell = st.T.data.shape[0]
    u_low = np.asarray(st.u.data)[:, -1]
    land = _make_mpas_turbulence(
        TurbulenceConfig(scheme=scheme), 300.0, f_land=jnp.ones((ncell,)))
    base_f = _land_forcing(ncell, 0.0, 0.0)
    bulk = _du_low(land(st, mpas_mesh, sigma_coord, forcing=base_f))
    strong = _du_low(land(st, mpas_mesh, sigma_coord,
                          forcing=_stress_forcing(ncell, 5.0)))
    assert np.sum(u_low * strong) < np.sum(u_low * bulk) < 0.0, (
        f"{scheme}: a 5 Pa land stress must remove lowest-level kinetic "
        "energy faster than the bulk stress")
    ocean = _make_mpas_turbulence(
        TurbulenceConfig(scheme=scheme), 300.0, f_land=jnp.zeros((ncell,)))
    o_bulk = _du_low(ocean(st, mpas_mesh, sigma_coord, forcing=base_f))
    o_with = _du_low(ocean(st, mpas_mesh, sigma_coord,
                           forcing=_stress_forcing(ncell, 5.0)))
    np.testing.assert_allclose(o_with, o_bulk, rtol=1e-10, atol=1e-14)


def test_land_stress_seed_follows_the_static_land_roughness(
        mpas_mesh, sigma_coord, mpas_state):
    """Before any valid land solve the drag is the neutral law of (z0m, d) at
    the lowest level's height above the surface: rougher or more displaced
    land must remove more lowest-level kinetic energy; a displacement far
    above the lowest level hits the clamp (identical for any larger d)."""
    st = _windy(mpas_state)
    ncell = st.T.data.shape[0]
    u_low = np.asarray(st.u.data)[:, -1]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme="louis"), 300.0, f_land=jnp.ones((ncell,)))

    def ke(z0m, d):
        f = dict(_stress_forcing(ncell, 0.0, valid=False),
                 z0m_land=jnp.full((ncell,), z0m), d_land=jnp.full((ncell,), d))
        return np.sum(u_low * _du_low(fn(st, mpas_mesh, sigma_coord,
                                         forcing=f)))
    assert ke(1.0, 0.0) < ke(0.01, 0.0) < 0.0
    assert ke(0.1, 5.0) < ke(0.1, 0.0)
    assert ke(0.1, 1.0e5) == ke(0.1, 2.0e5)
    assert ke(0.1, 1.0e5) < ke(0.1, 5.0)


def test_land_stress_with_partial_keys_is_refused(
        mpas_mesh, sigma_coord, mpas_state):
    ncell = mpas_state.T.data.shape[0]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme="louis"), 300.0, f_land=jnp.ones((ncell,)))
    f = _stress_forcing(ncell, 1.0)
    del f["z0m_land"]
    with pytest.raises(ValueError, match="z0m_land"):
        fn(mpas_state, mpas_mesh, sigma_coord, forcing=f)


def test_land_stress_without_land_heat_fluxes_is_refused(
        mpas_mesh, sigma_coord, mpas_state):
    ncell = mpas_state.T.data.shape[0]
    fn = _make_mpas_turbulence(
        TurbulenceConfig(scheme="louis"), 300.0, f_land=jnp.ones((ncell,)))
    with pytest.raises(ValueError, match="taumag_land"):
        fn(mpas_state, mpas_mesh, sigma_coord, forcing=dict(
            _land_forcing(ncell), taumag_land=jnp.ones((ncell,))))


def _eligible_cfg(**kw):
    kw.setdefault("use_multilayer_land", True)
    kw.setdefault("mpas_land_beta_soil", True)
    kw.setdefault("turbulence", "clubb")
    kw.setdefault("land_surface_scheme", "two_leaf")
    return _mpas_cfg(**kw)


@pytest.mark.parametrize("kw,why", [
    (dict(), None),
    (dict(land_surface_scheme="simple_seb"), None),
    (dict(use_multilayer_land=False, mpas_land_beta_soil=False),
     "use_multilayer_land"),
    (dict(mpas_land_beta_soil=False), "mpas_land_beta_soil"),
    (dict(turbulence="ysu"), "turbulence"),
    (dict(land_surface_scheme="clm_ml"), "land_surface_scheme"),
])
def test_land_stress_auto_resolution_per_predicate(kw, why):
    """AUTO (None) is ON exactly where eligible and OFF for each failing
    predicate (validation unchanged); explicit True is refused there; explicit
    False is always accepted and resolves off."""
    from legoesm.driver.config import (
        mpas_land_stress_eligibility, resolve_mpas_land_stress_from_land,
    )
    cfg = _eligible_cfg(**kw)
    ok, reason = mpas_land_stress_eligibility(cfg)
    assert ok is (why is None), reason
    assert resolve_mpas_land_stress_from_land(cfg) is ok
    assert resolve_mpas_land_stress_from_land(
        cfg._replace(mpas_land_stress_from_land=False)) is False
    if why is not None:
        assert why in reason
        with pytest.raises(ValueError, match="mpas_land_stress_from_land=True"):
            cfg._replace(mpas_land_stress_from_land=True).validate_strict()


def test_land_stress_auto_off_and_true_refused_off_the_mpas_lane():
    from legoesm.driver.config import resolve_mpas_land_stress_from_land
    cfg = _cdgrid_cfg()
    assert resolve_mpas_land_stress_from_land(cfg) is False
    cfg.validate_strict()
    with pytest.raises(ValueError, match="not the MPAS lane"):
        cfg._replace(mpas_land_stress_from_land=True).validate_strict()


def test_land_stress_lane_predicate_is_the_driver_dispatch():
    """The driver runs the MPAS lane on grid_type == 'mpas' (fv3_duo first);
    an mpas discretization on another grid never reaches it."""
    from legoesm.driver.config import mpas_land_stress_eligibility
    cfg = _eligible_cfg()
    assert mpas_land_stress_eligibility(cfg)[0]
    for bad in (cfg._replace(grid=cfg.grid._replace(grid_type="cubed_sphere")),
                cfg._replace(dycore=cfg.dycore._replace(
                    discretization="fv3_duo")),
                # the duo COLUMN lane runs the MPAS loop (mpas_loop_lane)
                # but is held out by name (not enabled or validated there)
                cfg._replace(dycore=cfg.dycore._replace(
                    discretization="fv3_duo", fv3_duo_column_lane=True))):
        ok, why = mpas_land_stress_eligibility(bad)
        assert not ok and "not the MPAS lane" in why


# ---------------------------------------------------------------------------
# Non-land fraction on the ocean surface (mpas_ocean_flux_on_ocean_surface,
# #1320 stage 1)
# ---------------------------------------------------------------------------

_T_OCEAN = 304.0     # warm ocean/ice surface [K]
_T_SKIN = 268.0      # cold land skin [K], blended into T_sfc over the land share


def _ocean_sfc_forcing(ncell, f_land, *, with_key=True, T_blend=None):
    """Land-flux handoff forcing over a mixed mesh; T_sfc is the land-blended
    anchor, T_sfc_ocean (optional) the non-land surface."""
    T_oc = jnp.full((ncell,), _T_OCEAN)
    if T_blend is None:
        T_blend = (1.0 - f_land) * T_oc + f_land * _T_SKIN
    f = _land_forcing(ncell, 40.0, 90.0)
    f["T_sfc"] = T_blend
    f["q_sfc_land"] = jnp.full((ncell,), 0.004)
    if with_key:
        f["T_sfc_ocean"] = T_oc
    return f


@pytest.fixture(scope="module")
def state64(mpas_state):
    """The fixture state in float64: the comparisons below are exact, and the
    fp32 state would make each run's output dtype follow its inputs' dtype."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64)
        if getattr(x, "dtype", None) == jnp.float32 else x, mpas_state)


def _fluxes(out):
    out = out[0] if isinstance(out, tuple) else out
    return {k: np.asarray(getattr(out, k).data)
            for k in ("shflx_sfc", "lhflx_sfc", "evap_sfc")}, out


@pytest.mark.parametrize("scheme", _LAND_FLUX_ACCEPTED)
def test_non_land_fraction_sees_the_ocean_surface(mpas_mesh, sigma_coord,
                                                  state64, scheme):
    """With T_sfc_ocean, the non-land share is the bulk flux of the OCEAN
    surface: it equals a run whose only surface IS that ocean surface (same
    land fluxes), whatever the land skin blended into T_sfc.  Tendencies too:
    with the flux injected the kernel must not re-read the blended T_sfc/q_sfc
    (if it did, the boundary state and the handed flux would disagree)."""
    ncell = state64.T.data.shape[0]
    f_land = jnp.asarray(np.linspace(0.1, 0.9, ncell))
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme=scheme), 300.0,
                               f_land=f_land)
    on = fn(state64, mpas_mesh, sigma_coord,
            forcing=_ocean_sfc_forcing(ncell, f_land))
    ref_f = _ocean_sfc_forcing(ncell, f_land, with_key=False,
                               T_blend=jnp.full((ncell,), _T_OCEAN))
    del ref_f["q_sfc_land"]                     # q_sfc = q_sat(T_ocean)
    ref = fn(state64, mpas_mesh, sigma_coord, forcing=ref_f)
    off = fn(state64, mpas_mesh, sigma_coord,
             forcing=_ocean_sfc_forcing(ncell, f_land, with_key=False))
    fl_on, t_on = _fluxes(on)
    fl_ref, t_ref = _fluxes(ref)
    fl_off, _ = _fluxes(off)
    for k in fl_on:
        np.testing.assert_allclose(fl_on[k], fl_ref[k], rtol=1e-12, atol=0,
                                   err_msg=k)
        assert not np.allclose(fl_on[k], fl_off[k]), (
            f"{k}: the ocean-surface key changed nothing (vacuous)")
    for name, a, b in (
            ("dT_dt", t_on.dT_dt.data, t_ref.dT_dt.data),
            ("du_dt", t_on.du_dt.data, t_ref.du_dt.data),
            ("q_v", t_on.tracer_tendencies["q_v"],
             t_ref.tracer_tendencies["q_v"])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12,
                                   atol=0, err_msg=name)
    # Turbulent carry (CLUBB/clubb_lite energy, cloud fraction): same.
    c_on, c_ref = on[1], ref[1]
    if c_on is not None:
        for a, b in zip(jax.tree_util.tree_leaves(c_on),
                        jax.tree_util.tree_leaves(c_ref)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                       rtol=1e-12, atol=0, err_msg="carry")


def test_non_land_closure_and_sign(mpas_mesh, sigma_coord, state64):
    """Closure: the handed flux is exactly (1-f)*ocean + f*land, so changing
    the land flux by d moves the total by f*d.  Sign: a 304 K ocean under
    290 K air is a SENSIBLE SOURCE; diluting it with the cold land skin (the
    old path) under-estimates it on every mixed cell."""
    ncell = state64.T.data.shape[0]
    f_land = jnp.asarray(np.linspace(0.1, 0.9, ncell))
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0,
                               f_land=f_land)
    f1 = _ocean_sfc_forcing(ncell, f_land)
    f2 = dict(f1, shflx_land=f1["shflx_land"] + 100.0)
    a, _ = _fluxes(fn(state64, mpas_mesh, sigma_coord, forcing=f1))
    b, _ = _fluxes(fn(state64, mpas_mesh, sigma_coord, forcing=f2))
    np.testing.assert_allclose(b["shflx_sfc"] - a["shflx_sfc"],
                               100.0 * np.asarray(f_land), rtol=1e-10)
    # Latent heat and water, which the inverse with L(T_sfc_ocean) feeds.
    f3 = dict(f1, lhflx_land=f1["lhflx_land"] + 100.0,
              evap_land=f1["evap_land"] + 1e-5)
    c, _ = _fluxes(fn(state64, mpas_mesh, sigma_coord, forcing=f3))
    np.testing.assert_allclose(c["lhflx_sfc"] - a["lhflx_sfc"],
                               100.0 * np.asarray(f_land), rtol=1e-10)
    np.testing.assert_allclose(c["evap_sfc"] - a["evap_sfc"],
                               1e-5 * np.asarray(f_land), rtol=1e-8)
    # The non-land water is the non-land latent heat over L(T_sfc_ocean).
    from legoesm.thermo import charged_latent_heat
    fl = np.asarray(f_land)
    lh_oc = (a["lhflx_sfc"] - fl * 90.0) / (1.0 - fl)
    ev_oc = (a["evap_sfc"] - fl * 90.0 / constants.L_v) / (1.0 - fl)
    from legoesm.atmosphere.physics.turbulence.integration import (
        get_turbulence_fn,
    )
    _bulk = get_turbulence_fn(TurbulenceConfig(scheme="louis"))[2].surface
    L_oc = np.asarray(charged_latent_heat(_bulk.bulk_scheme,
                                          jnp.full((ncell,), _T_OCEAN)))
    np.testing.assert_allclose(ev_oc, lh_oc / L_oc, rtol=1e-9)
    sh_ocean = (a["shflx_sfc"] - np.asarray(f_land) * 40.0) / (
        1.0 - np.asarray(f_land))
    assert (sh_ocean > 0.0).all(), "warm ocean under cold air must heat it"
    old, _ = _fluxes(fn(state64, mpas_mesh, sigma_coord,
                        forcing=_ocean_sfc_forcing(ncell, f_land,
                                                   with_key=False)))
    assert (a["shflx_sfc"] > old["shflx_sfc"]).all()


def test_ocean_only_cells_bit_identical(mpas_mesh, sigma_coord, state64):
    """Where f_land = 0 the driver's T_sfc_ocean equals T_sfc bit for bit, so
    the key must change nothing there."""
    ncell = state64.T.data.shape[0]
    f_land = jnp.asarray((np.arange(ncell) % 2) * 0.6)
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0,
                               f_land=f_land)
    on = fn(state64, mpas_mesh, sigma_coord,
            forcing=_ocean_sfc_forcing(ncell, f_land))
    off = fn(state64, mpas_mesh, sigma_coord,
             forcing=_ocean_sfc_forcing(ncell, f_land, with_key=False))
    ocean = np.asarray(f_land) == 0.0
    fo, to = _fluxes(on)
    ff, tf = _fluxes(off)
    for k in fo:
        np.testing.assert_array_equal(fo[k][ocean], ff[k][ocean], err_msg=k)
    np.testing.assert_array_equal(np.asarray(to.dT_dt.data)[ocean],
                                  np.asarray(tf.dT_dt.data)[ocean])


def test_ocean_surface_key_without_land_fluxes_is_refused(mpas_mesh,
                                                          sigma_coord,
                                                          state64):
    ncell = state64.T.data.shape[0]
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0,
                               f_land=jnp.full((ncell,), 0.5))
    f = {"T_sfc": jnp.full((ncell,), 300.0),
         "T_sfc_ocean": jnp.full((ncell,), 300.0)}
    with pytest.raises(ValueError, match="T_sfc_ocean"):
        fn(state64, mpas_mesh, sigma_coord, forcing=f)


def test_ocean_surface_gradient_finite_and_jit_parity(mpas_mesh, sigma_coord,
                                                     state64):
    ncell = state64.T.data.shape[0]
    f_land = jnp.asarray(np.linspace(0.1, 0.9, ncell))
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="louis"), 300.0,
                               f_land=f_land)
    base = _ocean_sfc_forcing(ncell, f_land)

    def loss(T_oc):
        out = fn(state64, mpas_mesh, sigma_coord,
                 forcing=dict(base, T_sfc_ocean=T_oc))
        out = out[0] if isinstance(out, tuple) else out
        return jnp.sum(out.dT_dt.data[:, -1] ** 2)

    g = jax.grad(loss)(base["T_sfc_ocean"])
    assert np.isfinite(np.asarray(g)).all() and np.abs(np.asarray(g)).max() > 0
    np.testing.assert_allclose(float(jax.jit(loss)(base["T_sfc_ocean"])),
                               float(loss(base["T_sfc_ocean"])), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(jax.jit(jax.grad(loss))(
        base["T_sfc_ocean"])), np.asarray(g), rtol=1e-8, atol=1e-30)


@pytest.mark.parametrize("kw,why", [
    (dict(use_multilayer_land=False, mpas_land_beta_soil=False),
     "use_multilayer_land"),
    (dict(mpas_land_beta_soil=False), "mpas_land_beta_soil"),
    (dict(turbulence="ysu"), "turbulence"),
])
def test_ocean_surface_flag_refused_where_inert(kw, why):
    cfg = _eligible_cfg(mpas_ocean_flux_on_ocean_surface=True)
    cfg.validate_strict()
    bad = cfg._replace(**kw)
    with pytest.raises(ValueError, match="mpas_ocean_flux_on_ocean_surface"):
        bad.validate_strict()
    from legoesm.driver.config import mpas_land_flux_handoff_eligibility
    ok, reason = mpas_land_flux_handoff_eligibility(bad)
    assert not ok and why in reason


def test_ocean_surface_flag_refused_without_sst_or_land_stress():
    cfg = _eligible_cfg(mpas_ocean_flux_on_ocean_surface=True)
    with pytest.raises(ValueError, match="radiation='none'"):
        cfg._replace(radiation="none").validate_strict()
    with pytest.raises(ValueError, match="land model's stress"):
        cfg._replace(mpas_land_stress_from_land=False).validate_strict()
    with pytest.raises(ValueError, match="land model's stress"):
        cfg._replace(land_surface_scheme="clm_ml").validate_strict()
    cfg._replace(mpas_land_stress_from_land=True).validate_strict()


def test_ocean_surface_flag_refused_off_the_mpas_lane():
    with pytest.raises(ValueError, match="not the MPAS lane"):
        _cdgrid_cfg(mpas_ocean_flux_on_ocean_surface=True).validate_strict()
