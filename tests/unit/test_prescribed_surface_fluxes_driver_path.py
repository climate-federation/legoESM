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
    class _S:
        sigma_full = jnp.linspace(0.1, 0.95, nlev)
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        dsigma = jnp.diff(jnp.linspace(0.05, 1.0, nlev + 1))
    return _S()


@pytest.fixture(scope="module", params=sorted(GRIDS))
def grid(request):
    return request.param, GRIDS[request.param]()


def _pipeline(grid, **cfg_overrides):
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False, **cfg_overrides)
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
