"""CAM6 cloud macrophysics on the spectral (WeatherBench training) lane.

The AMIP production configuration selects ``clouds: cam6_clubb``.  On the
hydrostatic and MPAS lanes radiation reads the CLUBB PDF cloud fraction and
the Zhang-McFarlane deep-convection inputs (interface mass flux, in-cloud
water) from the lagged ``PhysicsState`` carry.  The spectral lane had the
WRITE side of that contract (CLUBB and ZM both publish) and no READ side, so
``cam6_clubb`` raised at the first radiation trace and the WeatherBench
classical arm could not run the production physics at all.

These tests pin the read side, one seam each: the radiation factory, the
combined dispatcher, the convection bridge, the training rollout and the
classical split-radiation factory.

Non-vacuity: every test here FAILS on the commit before the wiring
(938855bde): the factory refused ``use_clubb_cloud_fraction`` for
``spectral_pe``, the dispatcher never forwarded ``phys_state`` to radiation,
the convection bridge never published ``mass_flux_up``/``icwmr``, and the
rollout never handed the carry to the radiation callable.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    SpectralPEConfig,
    isothermal_rest_state_spectral,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig  # noqa: E402
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.convection import integration as convint  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.output import ConvectionOutput  # noqa: E402
from legoesm.atmosphere.physics.physics_state import init_physics_state  # noqa: E402
from legoesm.atmosphere.physics.radiation import integration as radint  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import (  # noqa: E402
    RadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.training.neural_gcm_spectral import spectral_rollout  # noqa: E402

_N_MAX = 8
_N_LEV = 6
_DT = 900.0


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    # cam6_clubb refuses a state with no explicit condensate (CAM6 has no
    # diagnostic condensate floor), so the fixture carries liquid and ice.
    tracers = {"q_v": jnp.full(shp, 5.0e-3),
               "q_c": jnp.full(shp, 1.0e-4),
               "q_i": jnp.full(shp, 2.0e-5)}
    state = isothermal_rest_state_spectral(
        grid, sigma, T_init=280.0, p_s_init=1.0e5, tracers=tracers)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    return grid, sigma, state, ncol


def _rad_cfg(use_clubb_cf=True, scheme="cam6_clubb", nested=True):
    return RadiationConfig(
        scheme="rrtmgp", cloud_scheme=scheme,
        cloud_config=CloudConfig(scheme=scheme) if nested else None,
        use_clubb_cloud_fraction=use_clubb_cf,
        rrtmgp=RRTMGPConfig(include_clouds=True),
        diurnal_cycle=False)


def _seeded_carry(ncol):
    ps = init_physics_state(ncol, _N_LEV, PhysicsConfig())
    return ps._replace(
        cloud_fraction=jnp.full((ncol, _N_LEV), 0.37),
        conv_mass_flux_up=jnp.full((ncol, _N_LEV + 1), 0.02),
        conv_icwmr=jnp.full((ncol, _N_LEV), 2.0e-4),
    )


class _Captured(Exception):
    """Raised by the cloud-call spy once it has recorded its kwargs, so the
    chain physics_fn -> radiation backend -> compute_cloud_properties is
    exercised without running the RRTMGP solve."""


def test_spectral_factory_accepts_the_flag_and_refuses_cam6_without_it():
    fn = radint.make_radiation_physics(_rad_cfg(), model_type="spectral_pe",
                                       use_clubb_cloud_fraction=True)
    assert fn._wants_phys_state_ro is True
    assert "phys_state" in inspect.signature(fn).parameters
    for nested in (True, False):
        with pytest.raises(ValueError, match="use_clubb_cloud_fraction=True"):
            radint.make_radiation_physics(
                _rad_cfg(use_clubb_cf=False, nested=nested),
                model_type="spectral_pe", use_clubb_cloud_fraction=False)
    # Off the routing, the RH cloud path is untouched: no marker at all.
    fn_off = radint.make_radiation_physics(
        _rad_cfg(use_clubb_cf=False, scheme="sundqvist"),
        model_type="spectral_pe", use_clubb_cloud_fraction=False)
    assert not getattr(fn_off, "_wants_phys_state_ro", False)
    # The lane refusal still stands where no read side exists.
    with pytest.raises(NotImplementedError, match="spectral_pe"):
        radint.make_radiation_physics(_rad_cfg(), model_type="nonhydrostatic",
                                      use_clubb_cloud_fraction=True)


def test_spectral_cam6_radiation_refuses_a_missing_carry(setup):
    """A cam6 radiation fn called with no PhysicsState has no cloud fraction to
    read.  It must say so itself (GLM B-2): the alternative is a downstream
    'got None' from the cloud call, or -- worse -- a caller like
    spectral_amip_rollout believing it ran."""
    grid, sigma, state, _ncol = setup
    fn = radint.make_radiation_physics(_rad_cfg(), model_type="spectral_pe",
                                       use_clubb_cloud_fraction=True)
    with pytest.raises(ValueError, match="without the physics carry"):
        fn(state, grid, sigma)


def _no_remat(monkeypatch):
    """The spectral RRTMGP core is wrapped in ``jax.checkpoint``, under which
    the spy would only ever see tracers.  The values are what make these
    tests non-vacuous (a zero-initialised carry would pass a shape check), so
    the remat is bypassed for the spy runs; it changes nothing the spy reads."""
    monkeypatch.setattr(radint.jax, "checkpoint", lambda f, **kw: f)


def test_spectral_radiation_hands_the_carries_to_the_cloud_call(setup, monkeypatch):
    grid, sigma, state, ncol = setup
    _no_remat(monkeypatch)
    ps = _seeded_carry(ncol)
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        raise _Captured()
    monkeypatch.setattr(radint, "compute_cloud_properties", spy)

    for nested in (True, False):
        seen.clear()
        fn = radint.make_radiation_physics(
            _rad_cfg(nested=nested), model_type="spectral_pe",
            use_clubb_cloud_fraction=True)
        with pytest.raises(_Captured):
            fn(state, grid, sigma, phys_state=ps)
        assert np.allclose(np.asarray(seen["cloud_fraction_override"]), 0.37)
        assert np.allclose(np.asarray(seen["conv_mass_flux_up"]), 0.02)
        assert np.allclose(np.asarray(seen["conv_icwmr"]), 2.0e-4)
        assert seen["lat"].shape == (ncol,)
        assert seen["p_half"].shape == (ncol, _N_LEV + 1)

    seen.clear()
    fn_off = radint.make_radiation_physics(
        _rad_cfg(use_clubb_cf=False, scheme="sundqvist"),
        model_type="spectral_pe", use_clubb_cloud_fraction=False)
    with pytest.raises(_Captured):
        fn_off(state, grid, sigma)
    assert seen["cloud_fraction_override"] is None
    assert seen["conv_mass_flux_up"] is None


def test_spectral_combined_forwards_the_carry_to_radiation(setup, monkeypatch):
    grid, sigma, state, ncol = setup
    _no_remat(monkeypatch)
    ps = _seeded_carry(ncol)
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        raise _Captured()
    monkeypatch.setattr(radint, "compute_cloud_properties", spy)

    # Radiation is the first module in the chain, so the spy fires before
    # CLUBB runs: what it sees is the carry the dispatcher forwarded.
    combined = make_physics(PhysicsConfig(
        radiation=_rad_cfg(),
        turbulence=TurbulenceConfig(scheme="clubb",
                                    clubb=CLUBBConfig(prognostic=True)),
    ), model_type="spectral_pe", dt=_DT)
    with pytest.raises(_Captured):
        combined(state, grid, sigma, phys_state=ps)
    assert np.allclose(np.asarray(seen["cloud_fraction_override"]), 0.37)
    assert np.allclose(np.asarray(seen["conv_mass_flux_up"]), 0.02)
    assert np.allclose(np.asarray(seen["conv_icwmr"]), 2.0e-4)

    # The producer gate: routing a cloud fraction nobody writes would read
    # the zero-initialised carry and clear every cloud.  Refused at build.
    with pytest.raises(ValueError, match="cloud-fraction-producing"):
        make_physics(PhysicsConfig(
            radiation=_rad_cfg(),
            turbulence=TurbulenceConfig(scheme="louis"),
        ), model_type="spectral_pe", dt=_DT)


def test_spectral_convection_bridge_publishes_the_deepcu_inputs(setup, monkeypatch):
    grid, sigma, state, ncol = setup
    z = jnp.zeros((ncol, _N_LEV))

    # Distinct value per (column, level) so the published arrays must come
    # back in exactly the scheme's own ordering, not merely the right shape.
    mf_pattern = jnp.arange(ncol * (_N_LEV + 1), dtype=jnp.float64).reshape(
        ncol, _N_LEV + 1) * 1.0e-3
    icwmr_pattern = jnp.arange(ncol * _N_LEV, dtype=jnp.float64).reshape(
        ncol, _N_LEV) * 1.0e-6

    def fake_scheme(*, T, q_v, p_full, p_half, dt, config):
        return ConvectionOutput(
            dT_dt=z, dq_v_dt=z, dq_c_conv_dt=z, cape=jnp.zeros((ncol,)),
            convective_mask=jnp.zeros((ncol,)),
            mass_flux_up=mf_pattern, icwmr=icwmr_pattern)
    # 'dca' takes the plain (non-prognostic) bridge branch, whose call
    # signature the fake matches.
    monkeypatch.setattr(convint, "_get_convection_fn",
                        lambda cfg: ("dca", fake_scheme, cfg.dca))
    fn = convint.make_convection_physics(ConvectionConfig(scheme="dca"),
                                         "spectral_pe", _DT)
    _tend, prog = fn(state, grid, sigma)
    assert isinstance(prog, dict)
    assert prog["conv_mass_flux_up"].shape == (ncol, _N_LEV + 1)
    np.testing.assert_array_equal(np.asarray(prog["conv_mass_flux_up"]),
                                  np.asarray(mf_pattern))
    np.testing.assert_array_equal(np.asarray(prog["conv_icwmr"]),
                                  np.asarray(icwmr_pattern))

    # Half a publication is refused, as on the hydrostatic bridge.
    def half_scheme(*, T, q_v, p_full, p_half, dt, config):
        return ConvectionOutput(
            dT_dt=z, dq_v_dt=z, dq_c_conv_dt=z, cape=jnp.zeros((ncol,)),
            convective_mask=jnp.zeros((ncol,)),
            mass_flux_up=jnp.full((ncol, _N_LEV + 1), 0.01), icwmr=None)
    monkeypatch.setattr(convint, "_get_convection_fn",
                        lambda cfg: ("dca", half_scheme, cfg.dca))
    fn_half = convint.make_convection_physics(ConvectionConfig(scheme="dca"),
                                              "spectral_pe", _DT)
    with pytest.raises(ValueError, match="published together"):
        fn_half(state, grid, sigma)


def _marked_physics(grid, sigma):
    """The stateful marker protocol the classical factory attaches, bolted
    onto a bare combined physics fn (same construction as the CLUBB carry
    test)."""
    cfg = PhysicsConfig(turbulence=TurbulenceConfig(
        scheme="clubb", clubb=CLUBBConfig(prognostic=True)))
    raw = make_physics(cfg, model_type="spectral_pe", dt=_DT)

    def stateless(state, grid_, sigma_):
        out = raw(state, grid_, sigma_)
        return out[0] if isinstance(out, tuple) else out

    def stateful(state, grid_, sigma_, phys_state, forcing=None):
        out = raw(state, grid_, sigma_, phys_state=phys_state)
        assert isinstance(out, tuple) and len(out) == 2
        return out

    marked = lambda s, g, sg: stateless(s, g, sg)  # noqa: E731
    marked.with_phys_state = stateful
    marked.init_phys_state = (
        lambda ncol_, nlev_, dtype=None: init_physics_state(
            ncol_, nlev_, cfg, dtype=dtype))

    # The stateless CONTROL: a closure with no carry at all (prognostic CLUBB
    # rightly refuses a markerless call, so it cannot serve as the control).
    raw_plain = make_physics(PhysicsConfig(turbulence=TurbulenceConfig(
        scheme="louis")), model_type="spectral_pe", dt=_DT)

    def plain(state, grid_, sigma_):
        out = raw_plain(state, grid_, sigma_)
        return out[0] if isinstance(out, tuple) else out
    return marked, plain


def test_spectral_rollout_hands_the_carry_to_a_radiation_fn_that_reads_it():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    # No tracers here so the fake radiation tendency and the combined
    # non-radiative tendency share one pytree structure (tracers=None).
    state = isothermal_rest_state_spectral(grid, sigma, T_init=280.0,
                                           p_s_init=1.0e5)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    marked, plain = _marked_physics(grid, sigma)
    seen = {}

    def fake_rad(s, grid_, sigma_, *, sim_time_seconds=0.0, forcing=None,
                 phys_state=None):
        seen["phys_state"] = phys_state
        return s._replace(
            vor_hat=s.vor_hat.replace(data=jnp.zeros_like(s.vor_hat.data)),
            div_hat=s.div_hat.replace(data=jnp.zeros_like(s.div_hat.data)),
            T_hat=s.T_hat.replace(data=jnp.zeros_like(s.T_hat.data)),
            lnps_hat=s.lnps_hat.replace(data=jnp.zeros_like(s.lnps_hat.data)),
            phis_hat=s.phis_hat.replace(data=jnp.zeros_like(s.phis_hat.data)),
        )

    pe_cfg = SpectralPEConfig(hyperdiff_coeff=1e14, time_integrator="ssp_rk3")
    seed = marked.init_phys_state(ncol, _N_LEV)
    seen.clear()
    _final, ps_out = spectral_rollout(
        state, marked, grid, sigma, pe_cfg, dt=_DT, n_steps=1,
        rad_physics_fn=fake_rad, rad_update_interval=1,
        phys_state_in=seed, return_phys_state=True)
    # The stateful path hands radiation the carry (the lagged PhysicsState),
    # including on the very first call, which the old code made BEFORE the
    # carry existed.
    assert seen["phys_state"] is not None
    assert tuple(seen["phys_state"].clubb_moments.shape) == (ncol, 15, _N_LEV + 1)
    assert tuple(ps_out.clubb_moments.shape) == (ncol, 15, _N_LEV + 1)

    # A stateless (markerless) caller passes nothing: the kwarg is omitted and
    # the fn's own default (None) is what radiation sees.
    seen.clear()
    spectral_rollout(state, plain, grid, sigma, pe_cfg, dt=_DT, n_steps=1,
                     rad_physics_fn=fake_rad, rad_update_interval=1)
    assert seen["phys_state"] is None


def test_the_carry_is_filled_before_the_first_radiation_call():
    """Owner decision 2026-09-24: a radiation fn that reads the carry must not
    see the zero seed for its first gating window.  The rollout runs one
    non-radiative physics pass on the initial state and keeps the WHOLE carry
    from it (owner decision 2026-09-24: a self-consistent carry over a
    surgical three-field fill).  Only for a radiation fn that advertises the
    read, and only for a fresh seed.

    Non-vacuity: a supersaturated column makes CLUBB diagnose a non-zero PDF
    cloud fraction in a single pass, so "first radiation call saw the seed"
    (all zeros) and "saw a filled carry" are distinguishable.  The CLUBB
    moments must ALSO have left their seed in the reading arm (the whole carry
    advanced once) and be identical to it in the non-reading arm (the gate
    held)."""
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    state = isothermal_rest_state_spectral(
        grid, sigma, T_init=280.0, p_s_init=1.0e5,
        tracers={"q_v": jnp.full(shp, 2.0e-2)})   # supersaturated low levels
    ncol = int(grid.n_lat) * int(grid.n_lon)
    marked, _plain = _marked_physics(grid, sigma)
    pe_cfg = SpectralPEConfig(hyperdiff_coeff=1e14, time_integrator="ssp_rk3")
    seed = marked.init_phys_state(ncol, _N_LEV)
    first = {}

    def _zero_tend(s):
        tr = None
        if s.tracers is not None:
            tr = {k: (f.replace(data=jnp.zeros_like(f.data))
                      if hasattr(f, "replace") else jnp.zeros_like(f))
                  for k, f in s.tracers.items()}
        return s._replace(
            vor_hat=s.vor_hat.replace(data=jnp.zeros_like(s.vor_hat.data)),
            div_hat=s.div_hat.replace(data=jnp.zeros_like(s.div_hat.data)),
            T_hat=s.T_hat.replace(data=jnp.zeros_like(s.T_hat.data)),
            lnps_hat=s.lnps_hat.replace(data=jnp.zeros_like(s.lnps_hat.data)),
            phis_hat=s.phis_hat.replace(data=jnp.zeros_like(s.phis_hat.data)),
            tracers=tr,
        )

    def _record(ps):
        if "cf" not in first:
            first["cf"] = ps.cloud_fraction
            first["moments"] = ps.clubb_moments
            first["mf"] = ps.conv_mass_flux_up

    def reading_rad(s, grid_, sigma_, *, sim_time_seconds=0.0, forcing=None,
                    phys_state=None):
        _record(phys_state)
        return _zero_tend(s)
    reading_rad._wants_phys_state_ro = True

    def non_reading_rad(s, grid_, sigma_, *, sim_time_seconds=0.0,
                        forcing=None, phys_state=None):
        _record(phys_state)
        return _zero_tend(s)

    m0 = np.asarray(seed.clubb_moments)
    cf0 = np.asarray(seed.cloud_fraction)
    assert np.all(cf0 == 0.0)
    # FRESH seed on purpose: no ``phys_state_in`` (a supplied carry is the
    # chained-segment case, tested separately below).
    for rad, expect_filled in ((reading_rad, True), (non_reading_rad, False)):
        first.clear()
        spectral_rollout(state, marked, grid, sigma, pe_cfg, dt=_DT, n_steps=1,
                         rad_physics_fn=rad, rad_update_interval=1,
                         return_phys_state=True)
        cf = np.asarray(jax.device_get(first["cf"]))
        mom = np.asarray(jax.device_get(first["moments"]))
        assert np.all(np.isfinite(cf)) and np.all(np.isfinite(mom))
        assert bool(np.any(cf > 0.0)) is expect_filled, (
            "reading rad saw the zero seed" if expect_filled
            else "non-reading rad got a filled carry: the gate leaked")
        # The whole carry advanced once in the reading arm, not at all in the
        # non-reading arm: this is what distinguishes the full pass from a
        # three-field fill, and proves the gate on the marker.
        assert (not np.allclose(mom, m0)) is expect_filled, (
            "reading rad saw seed moments: the pass did not keep the whole carry"
            if expect_filled else
            "non-reading rad got advanced moments: the gate leaked")
        # No convection scheme in this config: the deepcu inputs stay seeded.
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(first["mf"])), np.asarray(seed.conv_mass_flux_up))

    # Chained segment: a SUPPLIED carry is used as-is, no fill.  Mark it with a
    # value the physics would never publish and check it comes through.
    first.clear()
    tagged = seed._replace(cloud_fraction=jnp.full((ncol, _N_LEV), 0.123))
    spectral_rollout(state, marked, grid, sigma, pe_cfg, dt=_DT, n_steps=1,
                     rad_physics_fn=reading_rad, rad_update_interval=1,
                     phys_state_in=tagged, return_phys_state=True)
    np.testing.assert_array_equal(np.asarray(jax.device_get(first["cf"])), 0.123)


def _cam6_bundle(cloud, turbulence="clubb"):
    from legoesm.training.aimip_params import (
        AIMIPTrainableBundle,
        aimip_legacy_owned_fields,
        aimip_scheme_keys_for,
    )
    from legoesm.training.model_registry import build_variant
    from legoesm.training.param_collector import build_trainable_params
    schemes = dict(convection="zhang_mcfarlane", turbulence=turbulence,
                   gwd="mcfarlane", microphysics="morrison", cloud=cloud)
    active = aimip_scheme_keys_for(radiation="rrtmgp", **schemes)
    bundle = AIMIPTrainableBundle(
        classical=build_variant(
            "classical", nlev=_N_LEV,
            overrides={"spatial_surface": False, "spatial_init_std": 0.0,
                       "spatial_seed": 0}),
        schemes=build_trainable_params(
            active_scheme_keys=active, tier="extended",
            exclude=tuple(sorted(aimip_legacy_owned_fields(cloud_scheme=cloud)))))
    return bundle, schemes


@pytest.mark.parametrize("cloud, expect", [("cam6_clubb", True), ("sundqvist", False)])
def test_the_classical_factory_selects_the_routing_for_cam6(setup, cloud, expect):
    """The WB deck never names the flag; the factory derives it from the cloud
    scheme because cam6_clubb has exactly one legal value for it."""
    from legoesm.training.aimip_params import make_aimip_classical_spectral_physics
    grid, sigma, _state, _ncol = setup
    bundle, schemes = _cam6_bundle(cloud)
    non_rad, rad = make_aimip_classical_spectral_physics(
        bundle, grid, _DT, radiation="rrtmgp", split_rad=True,
        rad_update_interval_steps=6, rrtmgp_gpoint_batch_size=8,
        param_overrides={"atm.turb.CLUBBConfig": {"prognostic": True}},
        convection_scheme=schemes["convection"],
        turbulence_scheme=schemes["turbulence"], gwd_scheme=schemes["gwd"],
        microphysics_scheme=schemes["microphysics"],
        cloud_scheme=schemes["cloud"], surface_bulk_scheme="constant")
    assert rad.radiation_config.use_clubb_cloud_fraction is expect
    assert "phys_state" in inspect.signature(rad).parameters
    assert hasattr(non_rad, "with_phys_state")
    # The rollout keys the carry warm-up on this marker; the factory must
    # re-export it from the raw radiation builder onto the wrapper.
    assert bool(getattr(rad, "_wants_phys_state_ro", False)) is expect


def test_the_split_factory_refuses_cam6_without_a_producer(setup):
    """The split builds radiation and the non-radiative physics separately,
    so the combined wrapper's producer gate never runs there (codex,
    CONFIRMED: cam6_clubb + louis built with the routing on and a zero cloud
    fraction).  The factory must apply the same gate to the resolved pair."""
    from legoesm.training.aimip_params import make_aimip_classical_spectral_physics
    grid, sigma, _state, _ncol = setup
    # The trainable set is built for the SAME closure the factory is asked
    # for, so the only refusal left to fire is the producer gate.
    bundle, schemes = _cam6_bundle("cam6_clubb", turbulence="louis")
    with pytest.raises(ValueError, match="cloud-fraction-producing"):
        make_aimip_classical_spectral_physics(
            bundle, grid, _DT, radiation="rrtmgp", split_rad=True,
            rad_update_interval_steps=6, rrtmgp_gpoint_batch_size=8,
            convection_scheme=schemes["convection"],
            turbulence_scheme=schemes["turbulence"], gwd_scheme=schemes["gwd"],
            microphysics_scheme=schemes["microphysics"],
            cloud_scheme=schemes["cloud"], surface_bulk_scheme="constant")
