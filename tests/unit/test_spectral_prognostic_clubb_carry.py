"""The prognostic CLUBB carry must survive a spectral rollout, not reset.

WHY THIS FILE EXISTS. Prognostic CLUBB — the higher-order moments carried as
state — was refused outright on the spectral lane, on the grounds that the
driver dropped the returned PhysicsState and the moments would silently
re-initialise every step. That premise stopped being true: the model's own
integrate loop, the driver's spectral lane and the training rollout all feed the
returned state back in. The refusal was lifted, so this file holds the evidence
the refusal used to stand in for, and it fails if the carry is ever dropped
again: the moments must (a) change from their seed, and (b) depend on how many
steps ran, which they cannot if each step re-seeds them.

The production AMIP configuration runs this scheme in this mode, which is why
the WeatherBench classical arm needs it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    SpectralPEConfig,
    isothermal_rest_state_spectral,
)
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.physics_state import init_physics_state  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.integration import (  # noqa: E402
    make_turbulence_physics,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.training.neural_gcm_spectral import spectral_rollout  # noqa: E402

_N_MAX = 8
_N_LEV = 8
_DT = 900.0


def _turb_config():
    return TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True))


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    state = isothermal_rest_state_spectral(
        grid, sigma, T_init=280.0, p_s_init=1.0e5)
    # A resting isothermal column has no shear and no buoyancy gradient, so
    # every moment tendency would be zero and "the carry never changes" would
    # pass for the wrong reason. Give the wind field structure.
    u = state.vor_hat.data
    idx = int(np.argmax(np.asarray(grid.ls) == 2))
    state = state._replace(
        vor_hat=state.vor_hat.replace(data=u.at[idx, :].add(1e-5)))
    return grid, sigma, state


def test_the_spectral_lane_builds_prognostic_clubb():
    """The factory refusal is gone: this is the AMIP production mode."""
    fn = make_turbulence_physics(_turb_config(), "spectral_pe", dt=_DT)
    assert callable(fn)


def test_the_nonhydrostatic_lane_still_refuses():
    """Only the spectral premise changed; that driver still drops the state."""
    with pytest.raises(NotImplementedError, match="nonhydrostatic"):
        make_turbulence_physics(_turb_config(), "nonhydrostatic", dt=_DT)


def _physics_fn(grid, sigma):
    cfg = PhysicsConfig(turbulence=_turb_config())
    raw = make_physics(cfg, model_type="spectral_pe", dt=_DT)

    def stateless(state, grid_, sigma_):
        out = raw(state, grid_, sigma_)
        return out[0] if isinstance(out, tuple) else out

    def stateful(state, grid_, sigma_, phys_state, forcing=None):
        out = raw(state, grid_, sigma_, phys_state=phys_state)
        assert isinstance(out, tuple) and len(out) == 2
        return out

    ncol = int(grid.n_lat) * int(grid.n_lon)
    stateless.with_phys_state = stateful
    stateless.init_phys_state = (
        lambda ncol_, nlev_, dtype=None: init_physics_state(
            ncol_, nlev_, cfg, dtype=dtype))
    return stateless, ncol


def test_the_moments_evolve_and_remember_the_previous_step(setup):
    """Two trajectories that END at the same atmospheric state, differing only
    in whether the carry was threaded.

    An earlier version compared one step against two and called the difference
    memory; it is not, because the second step also sees a different
    atmospheric state, so the two differ under re-seeding as well (GLM). Here
    the re-seeded run is started from the threaded run's own step-1 state, so
    the ONLY difference between the two endpoints is whether step 2 inherited
    step 1's moments. It fails by construction if the carry is dropped.
    """
    grid, sigma, state = setup
    fn, ncol = _physics_fn(grid, sigma)
    seed = fn.init_phys_state(ncol, _N_LEV)
    cfg = SpectralPEConfig(hyperdiff_coeff=1e14, time_integrator="ssp_rk3")

    s1, ps1 = spectral_rollout(
        state, fn, grid, sigma, cfg, n_steps=1, dt=_DT,
        phys_state_in=seed, return_phys_state=True)
    _, ps2 = spectral_rollout(
        state, fn, grid, sigma, cfg, n_steps=2, dt=_DT,
        phys_state_in=seed, return_phys_state=True)
    # Same step-2 atmospheric state, but step 2 starts from the FLOOR moments.
    _, ps2_reseeded = spectral_rollout(
        s1, fn, grid, sigma, cfg, n_steps=1, dt=_DT,
        phys_state_in=seed, return_phys_state=True)

    m0 = np.asarray(seed.clubb_moments)
    m1 = np.asarray(ps1.clubb_moments)
    m2 = np.asarray(ps2.clubb_moments)
    m2r = np.asarray(ps2_reseeded.clubb_moments)
    assert m1.shape == (ncol, 15, _N_LEV + 1)
    assert np.all(np.isfinite(m2)) and np.all(np.isfinite(m2r))
    assert not np.allclose(m1, m0), "the moments never left their seed"
    assert not np.allclose(m2, m2r), (
        "threading the carry into step 2 changed nothing: the moments are "
        "being re-seeded every step, which is the defect the old refusal "
        "guarded against")


def test_a_markerless_caller_is_refused_rather_than_silently_reseeded(setup):
    """The failure the lifted refusal could have become.

    Prognostic CLUBB reached through a physics_fn with no threaded state would
    restart its moments from the floor every step and still produce a
    plausible-looking run. That now raises.
    """
    grid, sigma, state = setup
    fn, _ = _physics_fn(grid, sigma)
    with pytest.raises(ValueError, match="re-seed"):
        fn(state, grid, sigma)


def test_the_production_factory_attaches_the_markers(setup):
    """The load-bearing fact for the training lane (GLM).

    The rollout threads the carry only for a physics_fn carrying the markers.
    The tests above bolt them on by hand; this one asserts the REAL classical
    factory attaches them, and that its seeded state has the prognostic slot
    at the right shape.
    """
    from legoesm.training.aimip_params import (
        AIMIPTrainableBundle,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.model_registry import build_variant
    from legoesm.training.param_collector import build_trainable_params
    from legoesm.training.aimip_params import (
        aimip_legacy_owned_fields,
        aimip_scheme_keys_for,
    )

    grid, sigma, _state = setup
    schemes = dict(convection="bechtold", turbulence="clubb", gwd="mcfarlane",
                   microphysics="morrison", cloud="sundqvist")
    active = aimip_scheme_keys_for(radiation="rrtmgp", **schemes)
    bundle = AIMIPTrainableBundle(
        classical=build_variant(
            "classical", nlev=_N_LEV,
            overrides={"spatial_surface": False, "spatial_init_std": 0.0,
                       "spatial_seed": 0}),
        schemes=build_trainable_params(
            active_scheme_keys=active, tier="extended",
            exclude=tuple(sorted(aimip_legacy_owned_fields(
                cloud_scheme="sundqvist")))))
    non_rad, _rad = make_aimip_classical_spectral_physics(
        bundle, grid, _DT, radiation="rrtmgp", split_rad=True,
        rad_update_interval_steps=6, rrtmgp_gpoint_batch_size=8,
        param_overrides={"atm.turb.CLUBBConfig": {"prognostic": True}},
        convection_scheme=schemes["convection"],
        turbulence_scheme=schemes["turbulence"], gwd_scheme=schemes["gwd"],
        microphysics_scheme=schemes["microphysics"],
        cloud_scheme=schemes["cloud"], surface_bulk_scheme="constant")
    assert hasattr(non_rad, "with_phys_state")
    assert hasattr(non_rad, "init_phys_state")
    ncol = int(grid.n_lat) * int(grid.n_lon)
    ps = non_rad.init_phys_state(ncol, _N_LEV)
    assert ps.clubb_moments is not None
    assert tuple(ps.clubb_moments.shape) == (ncol, 15, _N_LEV + 1)


@pytest.mark.slow
def test_a_gradient_reaches_back_through_the_carry(setup):
    """Training needs the moments ON the differentiation path, not just alive.

    Marked slow: the reverse pass through two CLUBB steps takes ~8 minutes on
    a CPU. A carry that is stop-gradient'd or re-seeded inside the step would leave
    this derivative exactly zero while every forward assertion above still
    passed.
    """
    grid, sigma, state = setup
    fn, ncol = _physics_fn(grid, sigma)
    seed = fn.init_phys_state(ncol, _N_LEV)
    cfg = SpectralPEConfig(hyperdiff_coeff=1e14, time_integrator="ssp_rk3")

    def loss(moments):
        ps = seed._replace(clubb_moments=moments)
        final, _ = spectral_rollout(
            state, fn, grid, sigma, cfg, n_steps=2, dt=_DT,
            phys_state_in=ps, return_phys_state=True)
        return jnp.sum(jnp.abs(final.T_hat.data))

    g = np.asarray(jax.grad(loss)(seed.clubb_moments))
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) > 0.0, "no gradient reaches the prognostic carry"
