"""Prescribed surface fluxes on the SHARED driver path, every grid.

Pins that a prescribed sensible/latent flux and stress become the lower
boundary condition of whatever surface scheme runs (Louis, YSU, the bulk
path) on the lat-lon C-grid, the cubed sphere and the MPAS Voronoi mesh,
that the radiative boundary condition formed from LW_up / SW_up / SW_down
reproduces the explicit T/emissivity/albedo overrides, and that the segment
forcing carries the planes.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.integration import (
    fold_prescribed_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.louis import LouisConfig
from legoesm.driver.compiled_segments import (
    GRID_SHAPED_FORCING_FIELDS,
    pack_forcing,
)
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh

from legoesm import constants

NLEV = 4
GRIDS = {
    "cubed_sphere": lambda: create_cubed_sphere(4),
    "latlon": lambda: create_latlon_grid(8, 16),
    "mpas": lambda: create_voronoi_mesh(3, lloyd_iterations=8),
}


def _sigma(nlev=NLEV):
    # full levels at the layer midpoints, so the kernels' rho*dz weighting
    # and the budget test's dp/g agree
    class _S:
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
        dsigma = jnp.diff(sigma_half)

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
    return _S()


@pytest.fixture(scope="module", params=sorted(GRIDS))
def grid(request):
    return request.param, GRIDS[request.param]()


def _pipeline(grid, **cfg_overrides):
    kw = dict(radiation="gray", microphysics="none", diurnal_cycle=False)
    kw.update(cfg_overrides)
    cfg = ExperimentConfig(**kw)
    return build_physics_pipeline(grid, _sigma(), cfg)


def _inputs(pipe):
    ad = pipe.adapter
    s2 = ad.shape_2d
    s3 = (*s2, NLEV)
    return dict(
        T=jnp.full(s3, 270.0), p_s=jnp.full(s2, 1.0e5),
        q_v=jnp.full(s3, 0.003), q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        u=jnp.full(s3, 3.0), v=jnp.zeros(s3),
        sst=jnp.full(s2, 295.0), sic=jnp.zeros(s2), lat=jnp.full(s2, 0.4),
    ), s2


def _step(pipe, **overrides):
    inp, s2 = _inputs(pipe)
    held3 = jnp.zeros((*s2, NLEV))
    held2 = jnp.zeros(s2)
    return pipe.physics_step_no_rad(
        inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"],
        jnp.zeros((pipe.adapter.ncol,)),
        inp["u"], inp["v"], inp["sst"], inp["sic"], inp["lat"], 600.0,
        held3, held2, held2, held2, held2, held2, **overrides)


def _fluxes(s2, shf, tau_x):
    return dict(sfc_shflx_override=jnp.full(s2, shf),
                sfc_lhflx_override=jnp.zeros(s2),
                sfc_taux_override=jnp.full(s2, tau_x),
                sfc_tauy_override=jnp.zeros(s2))


# ---------------------------------------------------------------- fold

def test_fold_is_identity_without_fluxes_and_writes_the_four_slots():
    cfg = LouisConfig()
    assert fold_prescribed_surface_fluxes(cfg) is cfg
    out = fold_prescribed_surface_fluxes(
        cfg, shflx_w_m2=jnp.full(3, 31.0), lhflx_w_m2=jnp.full(3, -12.5),
        tau_x_pa=jnp.full(3, 0.03), tau_y_pa=jnp.full(3, -0.04))
    assert float(out.surface.prescribed_shflx_w_m2[0]) == 31.0
    assert float(out.surface.prescribed_lhflx_w_m2[0]) == -12.5
    assert float(out.surface.prescribed_tau_x_pa[0]) == 0.03
    assert float(out.surface.prescribed_tau_y_pa[0]) == -0.04
    assert cfg.surface.prescribed_shflx_w_m2 is None  # input not mutated


# ---------------------------------------------------------------- surface schemes

@pytest.mark.parametrize("turbulence", ["louis", "ysu"])
def test_prescribed_fluxes_are_the_kernels_lower_bc(grid, turbulence):
    """Heat: the kernel echoes the prescribed flux and the lowest level
    warms/cools with its sign.  Momentum: +stress accelerates, -stress
    decelerates, symmetrically about zero stress.  No refusal."""
    _name, g = grid
    pipe = _pipeline(g, turbulence=turbulence)
    _, s2 = _inputs(pipe)
    pos = _step(pipe, **_fluxes(s2, 50.0, 0.1))
    neg = _step(pipe, **_fluxes(s2, -50.0, -0.1))
    np.testing.assert_allclose(np.asarray(pos.shflx), 50.0)
    np.testing.assert_allclose(np.asarray(neg.shflx), -50.0)
    dT_pos = float(jnp.mean(pos.dT_dt[..., -1]))
    dT_neg = float(jnp.mean(neg.dT_dt[..., -1]))
    assert dT_pos > 0.0 > dT_neg
    du_pos = float(jnp.mean(pos.du_dt[..., -1]))
    du_neg = float(jnp.mean(neg.du_dt[..., -1]))
    assert du_pos > 0.0 > du_neg
    assert du_pos == pytest.approx(-du_neg, rel=1e-6)


def test_prescribed_stress_kick_on_the_bulk_path_is_exact(grid):
    _name, g = grid
    pipe = _pipeline(g, turbulence="none")
    inp, s2 = _inputs(pipe)
    out = _step(pipe, **_fluxes(s2, 0.0, 0.1))
    ref = _step(pipe, **_fluxes(s2, 0.0, 0.0))
    dp_low = inp["p_s"] * (pipe.sigma_half[-1] - pipe.sigma_half[-2])
    expected = constants.g * 0.1 / dp_low
    np.testing.assert_allclose(
        np.asarray(out.du_dt[..., -1] - ref.du_dt[..., -1]),
        np.asarray(expected), rtol=1e-6)
    assert float(jnp.max(jnp.abs(out.du_dt[..., :-1] - ref.du_dt[..., :-1]))) == 0.0


def test_one_stress_component_alone_is_refused(grid):
    _name, g = grid
    pipe = _pipeline(g, turbulence="louis")
    _, s2 = _inputs(pipe)
    with pytest.raises(ValueError, match="pair"):
        _step(pipe, sfc_taux_override=jnp.full(s2, 0.1))


# ---------------------------------------------------------------- radiation BC

def _rad(pipe, field, **kw):
    """One radiation-core output: 0 = heating, 1 = surface net SW, 2 = net LW."""
    inp, s2 = _inputs(pipe)
    out = pipe.compute_radiation_core(
        inp["T"], inp["p_s"], inp["q_v"], inp["sst"], inp["sic"], inp["lat"],
        jnp.zeros(s2), 80.0, 43200.0, jnp.zeros(0), constants.S_0, None, None,
        u=inp["u"], v=inp["v"], dt=600.0, **kw)
    return out[field]


def test_prescribed_lw_up_matches_t_override_with_emissivity_one(grid):
    _name, g = grid
    pipe = _pipeline(g)
    _, s2 = _inputs(pipe)
    T = 289.0
    for field in (0, 2):   # heating and surface net LW
        a = _rad(pipe, field, sfc_lw_up=jnp.full(s2, constants.sigma_sb * T ** 4))
        b = _rad(pipe, field, sfc_T_override=jnp.full(s2, T),
                 sfc_emissivity_override=jnp.ones(s2))
        c = _rad(pipe, field)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
        assert float(jnp.max(jnp.abs(a - c))) > 0.0


def test_prescribed_sw_pair_matches_albedo_above_threshold_only(grid):
    _name, g = grid
    pipe = _pipeline(g)
    _, s2 = _inputs(pipe)
    # surface net SW is the output the albedo moves (gray's SW heating is
    # albedo-independent by construction)
    bright = _rad(pipe, 1, sfc_sw_up=jnp.full(s2, 400.0),
                  sfc_sw_down=jnp.full(s2, 500.0))
    alb = _rad(pipe, 1, sfc_albedo_override=jnp.full(s2, 0.8))
    np.testing.assert_allclose(np.asarray(bright), np.asarray(alb), rtol=1e-6)
    dark = _rad(pipe, 1, sfc_sw_up=jnp.full(s2, 0.4), sfc_sw_down=jnp.full(s2, 0.5))
    none = _rad(pipe, 1)
    np.testing.assert_allclose(np.asarray(dark), np.asarray(none), rtol=1e-9)
    assert float(jnp.max(jnp.abs(alb - none))) > 1.0   # W/m2: a real control
    with pytest.raises(ValueError, match="together"):
        _rad(pipe, 1, sfc_sw_up=jnp.full(s2, 400.0))


# ---------------------------------------------------------------- forcing

def test_pack_forcing_carries_the_planes_and_defaults_to_none():
    s2 = (3, 4)
    base = dict(sst=jnp.full(s2, 290.0), sic=jnp.zeros(s2), day_of_year=1.0,
                seconds_of_day=0.0, solar_weights=jnp.ones(s2), s_0=1361.0,
                o3_vmr=jnp.zeros((12, 4)), aerosol_od=jnp.zeros((12, 4)))
    f0 = pack_forcing(**base)
    new = ("sfc_taux_override", "sfc_tauy_override", "sfc_lw_up",
           "sfc_sw_up", "sfc_sw_down", "land_frac")
    for k in new:
        assert getattr(f0, k) is None
        assert k in GRID_SHAPED_FORCING_FIELDS
    f1 = pack_forcing(**base, **{k: jnp.full(s2, i + 1.0)
                                 for i, k in enumerate(new)})
    for i, k in enumerate(new):
        assert float(getattr(f1, k)[0, 0]) == i + 1.0


# ---------------------------------------------------------------- no double count

@pytest.mark.parametrize("turbulence", ["louis", "ysu"])
def test_prescribed_flux_enters_the_column_exactly_once(grid, turbulence):
    """Column budget: the water and momentum the column gains per unit time
    equal the prescribed surface fluxes (implicit vertical diffusion is
    conservative), so a second bulk kick on top would double them."""
    _name, g = grid
    pipe = _pipeline(g, turbulence=turbulence, convection="none")
    inp, s2 = _inputs(pipe)
    lhf, tau = 80.0, 0.1
    out = _step(pipe, sfc_shflx_override=jnp.zeros(s2),
                sfc_lhflx_override=jnp.full(s2, lhf),
                sfc_taux_override=jnp.full(s2, tau), sfc_tauy_override=jnp.zeros(s2))
    dp = inp["p_s"][..., None] * (pipe.sigma_half[1:] - pipe.sigma_half[:-1])
    water = jnp.sum(out.dq_v_dt * dp, axis=-1) / constants.g   # kg m-2 s-1
    momentum = jnp.sum(out.du_dt * dp, axis=-1) / constants.g   # Pa
    # Heat-only override: the water is lhf / L_v(T_sfc) -- the inverse of the
    # bulk law's own temperature-dependent latent heat (sst = 295 K here), not
    # the constant (2 % apart).
    from legoesm.thermo import latent_heat_vaporization
    np.testing.assert_allclose(np.asarray(water),
                               lhf / float(latent_heat_vaporization(jnp.asarray(295.0))), rtol=1e-3)
    # With the coupler's water channel the column gains exactly that water.
    out_w = _step(pipe, sfc_shflx_override=jnp.zeros(s2), sfc_lhflx_override=jnp.full(s2, lhf),
                  sfc_evap_override=jnp.full(s2, 2.5e-5))
    water_w = jnp.sum(out_w.dq_v_dt * dp, axis=-1) / constants.g
    np.testing.assert_allclose(np.asarray(water_w), 2.5e-5, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(momentum), tau, rtol=1e-3)


def test_tiled_surface_keeps_its_stress_when_only_heat_is_prescribed():
    """The coupler prescribes heat only: the tiled surface's own stress must
    survive (it is not what the heat flux replaces)."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        prescribed_into_surface_flux,
    )
    tx, ty, sh, lh, us = (jnp.full(3, 0.07), jnp.full(3, -0.02),
                          jnp.full(3, 10.0), jnp.full(3, 20.0), jnp.full(3, 0.3))
    rho = jnp.full(3, 1.2)
    out = prescribed_into_surface_flux((tx, ty, sh, lh, us), rho,
                                       shflx=jnp.full(3, 55.0))
    assert float(out[2][0]) == 55.0 and float(out[3][0]) == 20.0
    assert float(out[0][0]) == 0.07 and float(out[4][0]) == 0.3
    out = prescribed_into_surface_flux((tx, ty, sh, lh, us), rho,
                                       tau_x=jnp.full(3, 0.3), tau_y=jnp.zeros(3))
    assert float(out[2][0]) == 10.0
    assert float(out[4][0]) == pytest.approx(float(jnp.sqrt(0.3 / 1.2)), rel=1e-6)


# ---------------------------------------------------------------- rrtmgp + land tile

def test_prescribed_lw_up_on_rrtmgp_matches_t_override_with_emissivity_one():
    """Gray ignores the emissivity it is handed; RRTMGP consumes it, so the
    emissivity-1 half of the LW identity is pinned here (cubed sphere)."""
    pipe = _pipeline(create_cubed_sphere(4), radiation="rrtmgp")
    _, s2 = _inputs(pipe)
    T = 289.0
    a = _rad(pipe, 0, sfc_lw_up=jnp.full(s2, constants.sigma_sb * T ** 4))
    b = _rad(pipe, 0, sfc_T_override=jnp.full(s2, T),
             sfc_emissivity_override=jnp.ones(s2))
    c = _rad(pipe, 0, sfc_T_override=jnp.full(s2, T),
             sfc_emissivity_override=jnp.full(s2, 0.9))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
    assert float(jnp.max(jnp.abs(a - c))) > 0.0


def test_tiled_land_surface_keeps_its_stress_under_a_prescribed_heat_flux():
    """The coupled driver prescribes heat only over a tiled land/ocean
    surface: the kernel must still feel the TILED stress (it is not what the
    heat flux replaces) while echoing the prescribed heat flux."""
    pipe = _pipeline(create_cubed_sphere(4), turbulence="louis")
    _, s2 = _inputs(pipe)
    pipe.f_land = jnp.full(s2, 1.0)
    pipe.albedo_land = jnp.full(s2, 0.2)
    pipe.surface_tiled = True
    inp, _ = _inputs(pipe)
    ref = _step(pipe, T_land=jnp.full(s2, 285.0))
    pres = _step(pipe, T_land=jnp.full(s2, 285.0),
                 sfc_shflx_override=jnp.full(s2, 40.0),
                 sfc_lhflx_override=jnp.full(s2, 5.0))
    np.testing.assert_allclose(np.asarray(pres.shflx), 40.0)
    np.testing.assert_allclose(np.asarray(pres.lhflx), 5.0)
    # momentum untouched: the tiled stress survives the heat prescription
    np.testing.assert_allclose(np.asarray(pres.du_dt), np.asarray(ref.du_dt),
                               rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------- MPAS production physics

def test_mpas_production_physics_honours_the_anchors_and_the_radiative_bc():
    """The MPAS production lane builds its physics with
    ``combined.make_physics(cfg, model_type="mpas")`` (not the segment path):
    the PhysicsState anchor slots must reach its turbulence bridge and the
    radiation forcing keys its (hydrostatic-shared) radiation factory."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.grids.vertical import create_sigma_coordinate

    mesh = create_voronoi_mesh(3, lloyd_iterations=8)
    sigma = create_sigma_coordinate(6)
    state = held_suarez_init_mpas(mesh, sigma)
    cfg = PhysicsConfig(turbulence=TurbulenceConfig(scheme="louis"),
                        radiation=RadiationConfig(scheme="gray",
                                                  diurnal_cycle=False))
    fn = make_physics(cfg, model_type="mpas", dt=600.0)
    ncol, nlev = int(state.T.data.shape[0]), int(state.T.data.shape[1])
    ps = init_physics_state(ncol, nlev, cfg)

    def run(ps_, forcing=None):
        out = fn(state, mesh, sigma, phys_state=ps_, forcing=forcing)
        return out[0] if isinstance(out, tuple) else out

    base = run(ps)
    hot = run(ps._replace(surface_shflx_override_w_m2=jnp.full(ncol, 200.0),
                          surface_lhflx_override_w_m2=jnp.zeros(ncol),
                          surface_tau_x_override_pa=jnp.zeros(ncol),
                          surface_tau_y_override_pa=jnp.zeros(ncol)))
    cold = run(ps._replace(surface_shflx_override_w_m2=jnp.full(ncol, -200.0),
                           surface_lhflx_override_w_m2=jnp.zeros(ncol),
                           surface_tau_x_override_pa=jnp.zeros(ncol),
                           surface_tau_y_override_pa=jnp.zeros(ncol)))
    dT_hot = float(jnp.mean(hot.dT_dt.data[:, -1] - base.dT_dt.data[:, -1]))
    dT_cold = float(jnp.mean(cold.dT_dt.data[:, -1] - base.dT_dt.data[:, -1]))
    assert dT_hot > 0.0 > dT_cold
    # radiative BC through the MPAS radiation factory: LW_up = sigma T^4
    # with a warmer T changes the radiative tendency
    a = run(ps, {"sfc_lw_up": jnp.full(ncol, constants.sigma_sb * 320.0 ** 4)})
    b = run(ps, {"sfc_lw_up": jnp.full(ncol, constants.sigma_sb * 260.0 ** 4)})
    assert float(jnp.max(jnp.abs(a.dT_dt.data - b.dT_dt.data))) > 0.0


def test_prescribed_heat_is_a_convection_surface_source_without_a_turbulence_scheme():
    """Bechtold's shallow closure needs a surface flux source; a prescribed
    heat pair is one even on the bulk path (turbulence='none')."""
    pipe = _pipeline(create_cubed_sphere(4), turbulence="none",
                     convection="bechtold",
                     bechtold_use_ifs_shallow_closure=True)
    _, s2 = _inputs(pipe)
    out = _step(pipe, sfc_shflx_override=jnp.full(s2, 60.0),
                sfc_lhflx_override=jnp.full(s2, 90.0))
    assert bool(jnp.all(jnp.isfinite(out.dT_dt)))
    # ... and without the prescribed pair the closure still refuses (the
    # bulk path has no surface config to draw the flux from)
    with pytest.raises(ValueError):
        _step(pipe)


@pytest.mark.parametrize("turbulence", ["louis", "clubb_lite"])
def test_tiled_water_is_folded_even_without_any_override(monkeypatch, turbulence):
    """The mosaic's per-tile-inverted water is the kernel's moisture BC
    whenever the surface is tiled -- not only when the coupler prescribes
    something.  A fold gated on the override branch discarded it on the plain
    tiled path (the ordinary AMIP step), and a stress-only override tripped
    the pair rule because the tiled heat was never passed alongside."""
    from legoesm.atmosphere.physics.turbulence import integration as integ
    seen = []
    real = integ.fold_prescribed_surface_fluxes

    def spy(cfg, **kw):
        seen.append(kw)
        return real(cfg, **kw)

    monkeypatch.setattr(integ, "fold_prescribed_surface_fluxes", spy)
    pipe = _pipeline(create_cubed_sphere(4), turbulence=turbulence)
    _, s2 = _inputs(pipe)
    pipe.f_land = jnp.full(s2, 0.5)
    pipe.albedo_land = jnp.full(s2, 0.2)
    pipe.surface_tiled = True
    # clubb_lite carries prognostic TKE; seed it so the kernel runs.
    extra = ({"tke": jnp.full((pipe.adapter.ncol, NLEV), 0.1)}
             if turbulence == "clubb_lite" else {})
    _step(pipe, T_land=jnp.full(s2, 285.0), **extra)
    assert len(seen) == 1, "no fold happened on the plain tiled path"
    kw = seen[-1]
    assert kw["evap_kg_m2_s"] is not None and kw["lhflx_w_m2"] is not None
    assert kw["shflx_w_m2"] is None and kw["tau_x_pa"] is None
    water = np.asarray(kw["evap_kg_m2_s"])
    lh = np.asarray(kw["lhflx_w_m2"])
    assert np.isfinite(water).all() and (np.abs(water) > 0).any()
    # It is the per-tile water, not the blended heat over one L at the
    # blended temperature (ocean 295 K / land 285 K, half each).
    from legoesm.thermo import latent_heat_vaporization as L
    assert np.max(np.abs(water / (lh / float(L(290.0))) - 1.0)) > 1e-5
    # A stress-only override on the tiled path must not trip the pair rule.
    seen.clear()
    out = _step(pipe, T_land=jnp.full(s2, 285.0),
                sfc_taux_override=jnp.full(s2, 0.05),
                sfc_tauy_override=jnp.zeros(s2), **extra)
    assert np.isfinite(np.asarray(out.dT_dt)).all()
    assert seen[-1]["evap_kg_m2_s"] is not None and seen[-1]["lhflx_w_m2"] is not None


def test_physics_output_carries_the_surface_water_flux():
    """``PhysicsOutput.evap_sfc`` is the water the surface actually lost, on
    the same basis as the reported ``lhflx``: over open ocean at one skin
    temperature it is lhflx / L_v(SST) (the kernel's own inverse), never the
    constant-L inverse."""
    from legoesm.thermo import latent_heat_vaporization
    for turbulence in ("louis", "none"):
        pipe = _pipeline(create_cubed_sphere(4), turbulence=turbulence)
        _, s2 = _inputs(pipe)
        out = _step(pipe)
        assert out.evap_sfc is not None and np.isfinite(np.asarray(out.evap_sfc)).all()
        expected = np.asarray(out.lhflx) / float(latent_heat_vaporization(295.0))
        np.testing.assert_allclose(np.asarray(out.evap_sfc), expected, rtol=1e-10)
        assert np.max(np.abs(np.asarray(out.evap_sfc) / (np.asarray(out.lhflx) / constants.L_v) - 1.0)) > 1e-3

    # Tiled surface: evap_sfc is the folded per-tile water (the kernel's own
    # moisture BC), which no single-temperature inverse of lhflx reproduces.
    from legoesm.atmosphere.physics.turbulence import integration as integ
    seen = []
    real = integ.fold_prescribed_surface_fluxes
    integ.fold_prescribed_surface_fluxes = lambda cfg, **kw: (seen.append(kw), real(cfg, **kw))[1]
    try:
        pipe = _pipeline(create_cubed_sphere(4), turbulence="louis")
        _, s2 = _inputs(pipe)
        pipe.f_land = jnp.full(s2, 0.5)
        pipe.albedo_land = jnp.full(s2, 0.2)
        pipe.surface_tiled = True
        out = _step(pipe, T_land=jnp.full(s2, 285.0))
    finally:
        integ.fold_prescribed_surface_fluxes = real
    folded = np.asarray(pipe.adapter.unflatten_2d(seen[-1]["evap_kg_m2_s"]))
    np.testing.assert_allclose(np.asarray(out.evap_sfc), folded, rtol=1e-12)
    lh = np.asarray(out.lhflx)
    for T in (295.0, 285.0, 290.0):
        assert np.max(np.abs(np.asarray(out.evap_sfc) / (lh / float(latent_heat_vaporization(T))) - 1.0)) > 1e-5
