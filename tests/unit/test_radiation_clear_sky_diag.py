"""Clear-sky TOA diagnostic (CMIP6 ``rsutcs`` / ``rlutcs``) — PRODUCER side.

Covers ``RadiationConfig.clear_sky_diag`` and the cloud-free second radiation
pass in ``_make_hydrostatic_radiation`` (the factory MPAS uses:
``_make_mpas_radiation`` is an ALIAS of it, asserted below), i.e. the
``HydrostaticTendencies.{sw_up_toa,lw_up_toa,sw_down_sfc,lw_down_sfc}
_clearsky`` channel that feeds ``_sfc_diag`` slots 12-15 — the CMIP6 clear-sky
QUARTET rsutcs/rlutcs (TOA outgoing) + rsdscs/rldscs (surface downwelling).
All four come from ONE cloud-free solve.

Definition under test (CMIP6): clear-sky = the SAME radiative transfer with
CLOUDS removed and everything else — gases, ozone, AEROSOL — retained.  So:
  * ``rsutcs <  rsut``  (clouds add reflection)  =>  SW_CRE = rsut - rsutcs > 0
  * ``rlutcs >  rlut``  (clouds trap OLR)        =>  LW_CRE = rlutcs - rlut > 0
  * ``rsdscs >  rsds``  (clouds shade the surface)
  * ``rldscs <  rlds``  (clouds emit downward LW from cloud base)
All four are asserted against a REAL RRTMGP solve, not a mock.

The gate is a STATIC Python bool read in the factory closure (never a traced
``jnp.where``), so the default-off path emits no extra radiation HLO at all —
pinned by the backend call-count tests.
"""

import types

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.atmosphere.physics.radiation import integration as rad_int
from legoesm.atmosphere.physics.radiation.config import (
    RadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import create_sigma_coordinate

NLEV = 10

# The clear-sky quartet in _sfc_diag slot order (12, 13, 14, 15) and its
# all-sky partner per slot — the pairing every sign assertion rests on.
CLEARSKY_FIELDS = ("sw_up_toa_clearsky", "lw_up_toa_clearsky",
                   "sw_down_sfc_clearsky", "lw_down_sfc_clearsky")
CLEARSKY_ALLSKY_PARTNER = {
    "sw_up_toa_clearsky": "sw_up_toa",      # rsutcs <-> rsut   (+up)
    "lw_up_toa_clearsky": "lw_up_toa",      # rlutcs <-> rlut   (+up)
    "sw_down_sfc_clearsky": "sw_down_sfc",  # rsdscs <-> rsds   (+down)
    "lw_down_sfc_clearsky": "lw_down_sfc",  # rldscs <-> rlds   (+down)
}


def test_modules_under_test_resolve_in_this_checkout():
    """Editable-install trap guard: a worktree run must exercise the
    worktree's modules, not the main checkout's installed copy.  Without
    this every assertion below could be green against unmodified code."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    from legoesm.driver import model_driver
    from legoesm.core import state as core_state
    for mod in (rad_int, model_driver, core_state):
        assert root in pathlib.Path(mod.__file__).resolve().parents, (
            f"{mod.__name__} resolved to {mod.__file__}, OUTSIDE {root}")


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 2, lloyd_iterations=10)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(NLEV)


def _fld(a, dims):
    return Field(data=jnp.asarray(a), name="x", dims=dims, units="1")


def _cloudy_state(mesh, sigma, q_c=2.0e-4):
    """A uniformly CLOUDY column set (non-zero ``q_c`` at every level) so the
    cloud radiative effect is unambiguously non-zero."""
    n = int(mesh.nCells)
    sig = np.asarray(sigma.sigma_full)
    T = 220.0 + 70.0 * sig[None, :] * np.ones((n, 1))
    return types.SimpleNamespace(
        u=_fld(np.zeros((int(mesh.nEdges), NLEV)), ("edge", "lev")),
        T=_fld(T, ("cell", "lev")),
        p_s=_fld(np.full(n, 1.0e5), ("cell",)),
        phis=_fld(np.zeros(n), ("cell",)),
        v=None,
        tracers={
            "q_v": _fld(np.full((n, NLEV), 3.0e-3), ("cell", "lev")),
            "q_c": _fld(np.full((n, NLEV), q_c), ("cell", "lev")),
            "q_i": _fld(np.zeros((n, NLEV)), ("cell", "lev")),
        },
    )


def _cfg(clear_sky_diag, scheme="rrtmgp", cloud_scheme="sundqvist"):
    return RadiationConfig(
        scheme=scheme,
        rrtmgp=RRTMGPConfig(include_clouds=(cloud_scheme != "none")),
        cloud_scheme=cloud_scheme,
        clear_sky_diag=clear_sky_diag,
    )


# ---------------------------------------------------------------------------
# Config + factory wiring
# ---------------------------------------------------------------------------


def test_clear_sky_diag_defaults_off():
    """A field that defaulted ON would silently double every existing run's
    radiation cost."""
    assert RadiationConfig._field_defaults["clear_sky_diag"] is False
    assert RadiationConfig().clear_sky_diag is False


def test_mpas_uses_the_hydrostatic_radiation_builder():
    """The MPAS lane's builder must BE the one carrying the clear-sky pass —
    otherwise these tests prove nothing about the production lane
    (attribution gate: name the symbol that runs)."""
    assert rad_int._make_mpas_radiation is rad_int._make_hydrostatic_radiation


def test_tendency_carries_the_clear_sky_slots_last():
    """Trailing optionals only: an existing positional constructor must be
    unaffected."""
    from legoesm.core.state import HydrostaticTendencies
    # The quartet, then the later-added clear-sky SURFACE upwelling SW
    # (CMOR rsuscs, _sfc_diag slot 16).  It is kept OUT of
    # ``CLEARSKY_FIELDS`` because that tuple drives the all-sky-partner
    # assertions and rsuscs's partner (rsus) is DERIVED by the collector,
    # not carried on the tendency.
    # ``precip_solid`` (CMOR prsn) is a later trailing optional again; the
    # invariant this test protects is "the clear-sky fields are trailing
    # optionals with None defaults", not that they are physically last.
    assert HydrostaticTendencies._fields[-6:-1] == (
        CLEARSKY_FIELDS + ("sw_up_sfc_clearsky",))
    assert HydrostaticTendencies._fields[-1] == "precip_solid"
    for _k in CLEARSKY_FIELDS + ("sw_up_sfc_clearsky", "precip_solid"):
        assert HydrostaticTendencies._field_defaults[_k] is None, _k


# ---------------------------------------------------------------------------
# Backend call counting — the static gate must add ZERO work when off
# ---------------------------------------------------------------------------


def _count_backend_calls(monkeypatch, cfg, mesh, sigma, state):
    """Run the radiation physics_fn with a counting wrapper around
    ``_call_radiation_backend`` and return ``(n_calls, configs, tend)``."""
    real = rad_int._call_radiation_backend
    seen = []

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("radiation_config", None))
        return real(*args, **kwargs)

    monkeypatch.setattr(rad_int, "_call_radiation_backend", _spy)
    fn = make_radiation_physics(cfg, "mpas")
    tend = fn(state, mesh, sigma)
    return len(seen), seen, tend


def test_off_runs_exactly_one_solve_and_leaves_slots_none(
        monkeypatch, mesh, sigma):
    state = _cloudy_state(mesh, sigma)
    n_calls, _cfgs, tend = _count_backend_calls(
        monkeypatch, _cfg(False), mesh, sigma, state)
    assert n_calls == 1, "clear_sky_diag=False must not add a radiation solve"
    for _k in CLEARSKY_FIELDS:
        assert getattr(tend, _k) is None, _k


def test_on_with_clouds_runs_a_second_cloud_free_solve(
        monkeypatch, mesh, sigma):
    state = _cloudy_state(mesh, sigma)
    n_calls, cfgs, tend = _count_backend_calls(
        monkeypatch, _cfg(True), mesh, sigma, state)
    assert n_calls == 2, (
        "clear_sky_diag=True with an active cloud scheme needs a SECOND, "
        "cloud-free radiation solve — RRTMGP returns fluxes only for the "
        "optical state it was given")
    # Pass 1 = all-sky (clouds on); pass 2 = the cloud-free twin.
    assert cfgs[0].cloud_scheme == "sundqvist"
    assert cfgs[1].cloud_scheme == "none"
    assert cfgs[1].cloud_config is None
    assert cfgs[1].clear_sky_diag is False, (
        "the twin must not itself request a clear-sky pass")
    # Everything that is NOT a cloud must be IDENTICAL — clear-sky removes
    # clouds only (gases, ozone, aerosol, orbit, solar all retained).
    for _k in ("scheme", "rrtmgp", "ozone", "orbit", "diurnal_cycle",
               "rce_fixed_cos_zenith"):
        assert getattr(cfgs[1], _k) == getattr(cfgs[0], _k), _k
    for _k in CLEARSKY_FIELDS:
        assert getattr(tend, _k) is not None, _k


@pytest.mark.parametrize("scheme,cloud_scheme", [
    ("rrtmgp", "none"),   # no cloud reaches the solver
    ("gray", "none"),     # gray returns before the cloud block entirely
])
def test_on_without_active_clouds_aliases_instead_of_solving_twice(
        monkeypatch, mesh, sigma, scheme, cloud_scheme):
    """With no radiatively active cloud the all-sky solve IS the clear-sky
    solve — the builder must alias it (exact, free) rather than pay for a
    provably identical second call."""
    state = _cloudy_state(mesh, sigma)
    n_calls, _cfgs, tend = _count_backend_calls(
        monkeypatch, _cfg(True, scheme=scheme, cloud_scheme=cloud_scheme),
        mesh, sigma, state)
    assert n_calls == 1, "aliasing must not run a second radiation solve"
    for _clr, _all in CLEARSKY_ALLSKY_PARTNER.items():
        np.testing.assert_array_equal(
            np.asarray(getattr(tend, _clr).data),
            np.asarray(getattr(tend, _all).data), err_msg=_clr)


# ---------------------------------------------------------------------------
# Physics: the cloud radiative effect must have the right SIGN and magnitude
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cloudy_pair(mesh, sigma):
    """(tend_off, tend_on) for the SAME cloudy state — one variable changed
    (clear_sky_diag), everything else byte-identical."""
    state = _cloudy_state(mesh, sigma)
    off = make_radiation_physics(_cfg(False), "mpas")(state, mesh, sigma)
    on = make_radiation_physics(_cfg(True), "mpas")(state, mesh, sigma)
    return off, on


def test_all_sky_fluxes_are_unchanged_by_enabling_the_diagnostic(cloudy_pair):
    """The diagnostic must be READ-ONLY: turning it on may not perturb the
    all-sky solve (or the heating that drives the model)."""
    off, on = cloudy_pair
    for _k in ("sw_up_toa", "lw_up_toa", "sw_down_toa", "sw_net_sfc",
               "lw_net_sfc", "sw_down_sfc", "lw_down_sfc", "dT_dt"):
        np.testing.assert_array_equal(
            np.asarray(getattr(on, _k).data),
            np.asarray(getattr(off, _k).data), err_msg=_k)


def test_clear_sky_shortwave_is_less_reflective_than_all_sky(cloudy_pair):
    """rsutcs < rsut over a cloudy column: removing cloud removes albedo.
    SW_CRE = rsut - rsutcs must be POSITIVE."""
    _off, on = cloudy_pair
    rsut = np.asarray(on.sw_up_toa.data)
    rsutcs = np.asarray(on.sw_up_toa_clearsky.data)
    assert np.all(np.isfinite(rsutcs))
    assert np.all(rsutcs >= 0.0), "outgoing SW cannot be negative"
    assert np.all(rsutcs < rsut), (
        f"SW_CRE must be positive over cloud; got rsut={rsut.mean():.2f} "
        f"rsutcs={rsutcs.mean():.2f}")


def test_clear_sky_longwave_emits_more_than_all_sky(cloudy_pair):
    """rlutcs > rlut over a cloudy column: removing cloud removes the
    greenhouse trapping.  LW_CRE = rlutcs - rlut must be POSITIVE."""
    _off, on = cloudy_pair
    rlut = np.asarray(on.lw_up_toa.data)
    rlutcs = np.asarray(on.lw_up_toa_clearsky.data)
    assert np.all(np.isfinite(rlutcs))
    assert np.all(rlutcs > 0.0)
    assert np.all(rlutcs > rlut), (
        f"LW_CRE must be positive over cloud; got rlut={rlut.mean():.2f} "
        f"rlutcs={rlutcs.mean():.2f}")


def test_clear_sky_surface_shortwave_is_brighter_than_all_sky(cloudy_pair):
    """rsdscs > rsds: removing cloud stops shading the surface.  Positive
    DOWN — the same orientation as its all-sky partner ``sw_down_sfc``."""
    _off, on = cloudy_pair
    rsds = np.asarray(on.sw_down_sfc.data)
    rsdscs = np.asarray(on.sw_down_sfc_clearsky.data)
    assert np.all(np.isfinite(rsdscs))
    assert np.all(rsdscs >= 0.0), "downwelling SW cannot be negative"
    assert np.all(rsdscs > rsds), (
        f"clear sky must let MORE sunlight reach the surface; got "
        f"rsds={rsds.mean():.2f} rsdscs={rsdscs.mean():.2f}")


def test_clear_sky_surface_longwave_is_dimmer_than_all_sky(cloudy_pair):
    """rldscs < rlds: cloud base emits downward LW, so removing it REDUCES
    the surface downwelling longwave — the OPPOSITE direction to the
    shortwave, which is exactly why the sign is worth pinning."""
    _off, on = cloudy_pair
    rlds = np.asarray(on.lw_down_sfc.data)
    rldscs = np.asarray(on.lw_down_sfc_clearsky.data)
    assert np.all(np.isfinite(rldscs))
    assert np.all(rldscs > 0.0)
    assert np.all(rldscs < rlds), (
        f"clear sky must emit LESS downward LW at the surface; got "
        f"rlds={rlds.mean():.2f} rldscs={rldscs.mean():.2f}")


def test_clear_sky_equals_an_INDEPENDENT_cloud_free_run(mesh, sigma):
    """The strongest available check, and the one that validates the
    instrument: the quartet produced by the second pass must equal, to the
    bit, the ALL-SKY output of a separately built genuinely cloud-free
    model (``cloud_scheme='none'``) on the SAME state.

    This is what a weaker "clear-sky doesn't track q_c" assertion misses: a
    twin that merely stopped receiving the q_c TRACER but kept a cloud
    SCHEME would still diagnose cloud from RH, be independent of q_c, and
    silently publish a partly-cloudy field as clear-sky.
    """
    state = _cloudy_state(mesh, sigma)
    on = make_radiation_physics(_cfg(True), "mpas")(state, mesh, sigma)
    # Independent reference: no cloud scheme at all, no clear-sky machinery.
    ref = make_radiation_physics(
        _cfg(False, cloud_scheme="none"), "mpas")(state, mesh, sigma)
    for _clr, _all in CLEARSKY_ALLSKY_PARTNER.items():
        np.testing.assert_allclose(
            np.asarray(getattr(on, _clr).data),
            np.asarray(getattr(ref, _all).data),
            rtol=1e-12, atol=1e-12, err_msg=(
                f"{_clr} does not match an independent cloud-free solve — "
                f"the 'clear-sky' pass still sees cloud"))
    # Control: the reference really IS radiatively different from all-sky,
    # so the equality above is a match, not a tautology.
    assert not np.allclose(np.asarray(ref.sw_up_toa.data),
                           np.asarray(on.sw_up_toa.data)), (
        "control failed: the cloud-free reference did not differ from "
        "all-sky, so this test proves nothing")


def test_clear_sky_is_insensitive_to_the_cloud_amount(mesh, sigma):
    """The decisive check that the second pass really is CLOUD-FREE: doubling
    q_c must move rsut/rlut but leave rsutcs/rlutcs bit-identical.  A pass
    that leaked cloud input would track the cloud amount."""
    fn = make_radiation_physics(_cfg(True), "mpas")
    thin = fn(_cloudy_state(mesh, sigma, q_c=1.0e-4), mesh, sigma)
    thick = fn(_cloudy_state(mesh, sigma, q_c=4.0e-4), mesh, sigma)
    # All-sky RESPONDS to the cloud (the control that proves the two states
    # really differ radiatively).
    assert not np.allclose(np.asarray(thin.sw_up_toa.data),
                           np.asarray(thick.sw_up_toa.data)), (
        "control failed: q_c change did not move the all-sky SW")
    # Clear-sky does NOT — for ANY member of the quartet.
    for _k in CLEARSKY_FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(thin, _k).data),
            np.asarray(getattr(thick, _k).data), err_msg=_k)


def test_clear_sky_fields_have_cmor_units_and_shape(cloudy_pair, mesh):
    _off, on = cloudy_pair
    n = int(mesh.nCells)
    for _clr, _all in CLEARSKY_ALLSKY_PARTNER.items():
        f = getattr(on, _clr)
        assert f.units == "W/m^2", _clr
        assert np.asarray(f.data).shape == (n,), _clr
        # Same 2-D cell dims as the all-sky partner it is published with.
        assert f.dims == getattr(on, _all).dims, _clr


# ---------------------------------------------------------------------------
# JIT parity (the repo's standing requirement for any new physics branch)
# ---------------------------------------------------------------------------


def test_jit_parity_of_the_clear_sky_pass(mesh, sigma):
    """Eager vs ``jit`` must agree: the gate is a static Python bool, so no
    tracer-dependent branching can have crept in."""
    state = _cloudy_state(mesh, sigma)
    fn = make_radiation_physics(_cfg(True), "mpas")
    eager = fn(state, mesh, sigma)

    def _run(T_data):
        s = types.SimpleNamespace(
            u=state.u, T=state.T.replace(data=T_data), p_s=state.p_s,
            phis=state.phis, v=None, tracers=state.tracers)
        t = fn(s, mesh, sigma)
        return tuple(getattr(t, _k).data for _k in CLEARSKY_FIELDS)

    jitted = jax.jit(_run)(state.T.data)
    for _k, _got in zip(CLEARSKY_FIELDS, jitted):
        np.testing.assert_allclose(
            np.asarray(_got), np.asarray(getattr(eager, _k).data),
            rtol=1e-10, atol=1e-10, err_msg=_k)
