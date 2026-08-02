"""Clear-sky TOA diagnostic (CMIP6 ``rsutcs`` / ``rlutcs``) — PRODUCER side.

Covers ``RadiationConfig.clear_sky_diag`` and the cloud-free second radiation
pass in ``_make_hydrostatic_radiation`` (the factory MPAS uses:
``_make_mpas_radiation`` is an ALIAS of it, asserted below), i.e. the
``HydrostaticTendencies.sw_up_toa_clearsky`` / ``lw_up_toa_clearsky`` channel
that feeds ``_sfc_diag`` slots 12/13.

Definition under test (CMIP6): clear-sky = the SAME radiative transfer with
CLOUDS removed and everything else — gases, ozone, AEROSOL — retained.  So:
  * ``rsutcs <= rsut``  (clouds add reflection)  =>  SW_CRE = rsut - rsutcs > 0
  * ``rlutcs >= rlut``  (clouds trap OLR)        =>  LW_CRE = rlutcs - rlut > 0
Both are asserted against a REAL RRTMGP solve, not a mock.

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
    assert HydrostaticTendencies._fields[-2:] == (
        "sw_up_toa_clearsky", "lw_up_toa_clearsky")
    assert HydrostaticTendencies._field_defaults[
        "sw_up_toa_clearsky"] is None
    assert HydrostaticTendencies._field_defaults[
        "lw_up_toa_clearsky"] is None


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
    assert tend.sw_up_toa_clearsky is None
    assert tend.lw_up_toa_clearsky is None


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
    assert tend.sw_up_toa_clearsky is not None
    assert tend.lw_up_toa_clearsky is not None


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
    np.testing.assert_array_equal(
        np.asarray(tend.sw_up_toa_clearsky.data),
        np.asarray(tend.sw_up_toa.data))
    np.testing.assert_array_equal(
        np.asarray(tend.lw_up_toa_clearsky.data),
        np.asarray(tend.lw_up_toa.data))


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
               "lw_net_sfc", "dT_dt"):
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
    # Clear-sky does NOT.
    np.testing.assert_array_equal(
        np.asarray(thin.sw_up_toa_clearsky.data),
        np.asarray(thick.sw_up_toa_clearsky.data))
    np.testing.assert_array_equal(
        np.asarray(thin.lw_up_toa_clearsky.data),
        np.asarray(thick.lw_up_toa_clearsky.data))


def test_clear_sky_fields_have_cmor_units_and_shape(cloudy_pair, mesh):
    _off, on = cloudy_pair
    n = int(mesh.nCells)
    for f in (on.sw_up_toa_clearsky, on.lw_up_toa_clearsky):
        assert f.units == "W/m^2"
        assert np.asarray(f.data).shape == (n,)
        # Same 2-D cell dims as the all-sky TOA pair they are published with.
        assert f.dims == on.sw_up_toa.dims


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
        return t.sw_up_toa_clearsky.data, t.lw_up_toa_clearsky.data

    sw_j, lw_j = jax.jit(_run)(state.T.data)
    np.testing.assert_allclose(
        np.asarray(sw_j), np.asarray(eager.sw_up_toa_clearsky.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(lw_j), np.asarray(eager.lw_up_toa_clearsky.data),
        rtol=1e-10, atol=1e-10)
