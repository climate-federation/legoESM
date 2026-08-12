"""Veros u_centered dzw slot for the implicit vertical-diffusion solves
(``config.implicit_vmix_dzw_slot``, #428).

The backward-Euler tracer (T/S) and momentum-friction vertical-diffusion solves
take a GRADIENT (center-to-center) divisor.  legoESM's default is the midpoint
reconstruction ``build_dz_half(dz_cell) = 0.5(dz_k + dz_{k+1})``; Veros uses the
coordinate's center-to-center spacing ``dzw`` (``thermodynamics.py:267
delta = dt·kappaH/dzw``), which legoESM carries as ``z_coord.dz_half_ref``.

On a midpoint z-star coordinate ``dz_half_ref == build_dz_half(dz_ref)`` so the
flag is a NO-OP.  On a u_centered z-coordinate (the Veros-faithful ACC recipe)
the two differ per level, so at identical diffusivity the discrete flux differs.

The trio of equivalence tests pins the wiring exactly:
  * flag OFF ignores ``dz_half_ref`` (depends only on ``dz_ref``);
  * flag ON uses ``dz_half_ref``;
  * the two agree iff ``dz_half_ref`` is the midpoint.

Constant background A_v/K_v (``physics=None`` ⇒ state-independent K profiles) so
the solve metric is the only thing under test.  Run in the fp64 policy.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_DT = 1800.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _channel(z_coord=None, n_lat=8, n_lon=16, **cfg_kw):
    """Closed channel with a thermal front and a vertically-sheared jet, on the
    supplied vertical coordinate (defaults to a midpoint z-star)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    if z_coord is None:
        z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=float(np.sum(np.asarray(z_coord.dz_ref))),
        land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    nlev = z_coord.n_levels
    shear = np.linspace(1.0, 0.1, nlev)[None, None, :]
    u = 0.2 * np.cos(np.radians(lat))[:, None, None] * shear
    u = np.broadcast_to(u, state.u.data.shape).copy()
    u *= np.asarray(state.u_mask.data)[..., None]
    u[:, -1] = u[:, 0]
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
    # Vertically-sheared meridional flow so the v friction solve is nontrivial
    # too (friction on a zero field is a slot-independent no-op).
    nv = state.v.data.shape[0]
    vlat = np.linspace(-1.0, 1.0, nv)[:, None, None]
    v = 0.1 * vlat * shear
    v = np.broadcast_to(v, state.v.data.shape).copy()
    v *= np.asarray(state.v_mask.data)[..., None]
    state = state._replace(v=state.v.replace(data=jnp.asarray(v)))
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg_kw.setdefault("outer_integrator", "ab2")
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _midpoint_zstar():
    from legoesm.ocean.vertical import create_ocean_z_star
    return create_ocean_z_star(n_levels=4, H_max=4000.0)


def _u_centered_like():
    """A z-star whose center-to-center spacing dz_half_ref is perturbed away
    from the midpoint (a controlled stand-in for the Veros u_centered grid:
    dz_half_ref != 0.5(dz_k+dz_{k+1}), dz_ref / interfaces unchanged)."""
    z = _midpoint_zstar()
    nlev = z.n_levels
    factors = jnp.asarray(
        [1.0 + 0.4 * ((-1.0) ** k) for k in range(nlev - 1)],
        dtype=z.dz_half_ref.dtype)          # alternating ±40%, like Veros dzw
    return z._replace(dz_half_ref=z.dz_half_ref * factors)


# --------------------------------------------------------------- premise

def test_midpoint_zstar_dz_half_ref_is_midpoint():
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    z = _midpoint_zstar()
    np.testing.assert_allclose(
        np.asarray(z.dz_half_ref),
        np.asarray(build_dz_half(z.dz_ref)), rtol=0, atol=1e-12)


def test_u_centered_dz_half_ref_differs_from_midpoint():
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    z = _u_centered_like()
    assert np.max(np.abs(np.asarray(z.dz_half_ref)
                         - np.asarray(build_dz_half(z.dz_ref)))) > 1.0


def test_acc_recipe_coord_is_u_centered():
    """The shipped ACC recipe coordinate is genuinely u_centered (the slot the
    flag targets in production)."""
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_z_coord
    z = build_acc_z_coord()
    assert np.max(np.abs(np.asarray(z.dz_half_ref)
                         - np.asarray(build_dz_half(z.dz_ref)))) > 1.0


# --------------------------------------------------------------- validation

def test_rejects_without_implicit_vmix():
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        _channel(implicit_vmix_dzw_slot=True, implicit_vertical_mixing=False)


# ----------------------------------------------------- slot semantics

def _solve(z_coord, flag):
    state, model = _channel(z_coord=z_coord, implicit_vmix_dzw_slot=flag)
    out = model._apply_implicit_vertical_mixing(state, _DT, None)
    return {k: np.asarray(getattr(out, k).data) for k in ("T", "S", "u", "v")}


def test_noop_on_midpoint_zstar():
    """Where dz_half_ref IS the midpoint, flag ON == OFF, bit-identical."""
    z = _midpoint_zstar()
    off, on = _solve(z, False), _solve(z, True)
    for k in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(on[k], off[k])


def test_active_on_u_centered():
    """Where dz_half_ref differs from the midpoint, flag ON changes every solved
    field (guards against a silently-dead flag)."""
    z = _u_centered_like()
    off, on = _solve(z, False), _solve(z, True)
    for k in ("T", "S", "u", "v"):
        assert np.max(np.abs(on[k] - off[k])) > 0.0, f"{k} unchanged by flag"


def test_flag_off_ignores_dz_half_ref():
    """The DEFAULT slot is build_dz_half(dz_cell), a function of dz_ref only —
    overriding dz_half_ref must not move the flag-OFF result."""
    z_mid, z_uc = _midpoint_zstar(), _u_centered_like()
    a, b = _solve(z_mid, False), _solve(z_uc, False)
    for k in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(a[k], b[k])


def test_flag_on_tracks_dz_half_ref():
    """The faithful slot IS dz_half_ref: flag-ON on the perturbed coordinate
    differs from flag-ON on the midpoint coordinate (same dz_ref, different
    dz_half_ref) — i.e. flag-ON consumes dz_half_ref, not the midpoint."""
    z_mid, z_uc = _midpoint_zstar(), _u_centered_like()
    a, b = _solve(z_mid, True), _solve(z_uc, True)
    for k in ("T", "S", "u", "v"):
        assert np.max(np.abs(a[k] - b[k])) > 0.0


# ------------------------------------------------------------ robustness

def test_stability_100_steps_u_centered():
    z = _u_centered_like()
    state, model = _channel(z_coord=z, implicit_vmix_dzw_slot=True)
    for _ in range(100):
        state = model.step(state, _DT)
    u = np.asarray(state.u.data)
    T = np.asarray(state.T.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(T))
    assert np.max(np.abs(u)) < 5.0
    assert -5.0 < T.min() and T.max() < 40.0


def test_differentiable():
    z = _u_centered_like()
    state, model = _channel(z_coord=z, n_lat=6, n_lon=8,
                            implicit_vmix_dzw_slot=True)

    def loss(scale):
        st = state._replace(u=state.u.replace(data=state.u.data * scale))
        s1 = model.step(st, _DT)
        s2 = model.step(s1, _DT)
        return jnp.sum(s2.u.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0


# --------------------------------------------------- NEMO e3w(Kmm) divisor
# (#1226 W1): trazdf.F90:219-220 divides the implicit flux coefficient by
# e3w(...,Kmm), called from stpmlf.F90:370 as tra_zdf(kstp,Nbb,Nnn,Nrhs,ts,Naa)
# -- the dummy arg Kmm binds to Nnn, NEMO's NOW time level. legoESM's default
# divisor uses the AFTER-solve (barotropic-updated) thickness; this option
# uses the NOW (pre-solve) thickness instead, threaded via the ``eta_now``
# kwarg exactly as production wires it in ``_unsplit_ab2_step`` (state_corr.eta
# is AFTER/Naa; eta_now=state.eta.data is the true pre-solve NOW/Nnn).

def test_rejects_e3t_now_without_implicit_vmix():
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        _channel(implicit_vmix_e3t_now_divisor=True,
                 implicit_vertical_mixing=False)


def test_e3t_now_and_dzw_slot_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _channel(implicit_vmix_dzw_slot=True,
                 implicit_vmix_e3t_now_divisor=True)


def test_e3t_now_divisor_defaults_to_state_eta_when_unthreaded():
    """Without an explicit eta_now (every call site except the unsplit-AB2
    one), the option falls back to state.eta -- bit-identical to a run where
    eta_now is passed but equals state.eta.data exactly (the NO eta-tendency
    case). Confirms the fallback wiring, not just its absence of a crash."""
    z = _midpoint_zstar()
    state, model = _channel(z_coord=z, implicit_vertical_mixing=True,
                            implicit_vmix_e3t_now_divisor=True)
    implicit = model._apply_implicit_vertical_mixing(state, _DT, None)
    explicit = model._apply_implicit_vertical_mixing(
        state, _DT, None, eta_now=state.eta.data)
    for k in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(implicit, k).data),
            np.asarray(getattr(explicit, k).data))


def test_e3t_now_divisor_noop_when_eta_now_equals_state_eta():
    """When eta_now == state.eta (no eta tendency between NOW and the AFTER
    state the function's dz_cell is built from -- e.g. a rest-state
    barotropic solve), the NEMO NOW-divisor and legoESM's default AFTER-
    divisor read the SAME thickness -> bit-identical, fp64."""
    z = _midpoint_zstar()
    state, model = _channel(z_coord=z, implicit_vertical_mixing=True)
    off = model._apply_implicit_vertical_mixing(state, _DT, None)
    _, model_on = _channel(
        z_coord=z, implicit_vertical_mixing=True,
        implicit_vmix_e3t_now_divisor=True)
    on = model_on._apply_implicit_vertical_mixing(
        state, _DT, None, eta_now=state.eta.data)
    for k in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(on, k).data), np.asarray(getattr(off, k).data))


def test_e3t_now_flag_flips_via_namedtuple_replace():
    """LatLonCGridOceanConfig is a NamedTuple: the flag is flipped with
    ``._replace``, NOT ``dataclasses.replace`` (which raises TypeError on a
    NamedTuple -- the exact footgun the physics-validator review caught in
    the first driver wiring)."""
    import dataclasses
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat(implicit_vertical_mixing=True)
    on = cfg._replace(implicit_vmix_e3t_now_divisor=True)
    assert on.implicit_vmix_e3t_now_divisor is True
    assert cfg.implicit_vmix_e3t_now_divisor is False
    with pytest.raises(TypeError):
        dataclasses.replace(cfg, implicit_vmix_e3t_now_divisor=True)


def test_e3t_now_divisor_active_on_leapfrog_step():
    """The leapfrog-MLF step (the DINO kamm_mlf production integrator) must
    THREAD the NOW eta into the implicit solve: without the eta_now threading
    at _leapfrog_step's call site, the flag's fallback would read
    naa_expl.eta -- the SAME AFTER-level eta the default divisor is built
    from -- making ON bit-identical to OFF on a midpoint z-star. So ON != OFF
    after leapfrog steps with evolving eta proves the threading exists."""
    z = _midpoint_zstar()
    # leapfrog's own config validation requires explicit_ab2 Coriolis.
    lf = dict(outer_integrator="leapfrog", coriolis_scheme="explicit_ab2",
              implicit_vertical_mixing=True)
    state_off, model_off = _channel(z_coord=z, **lf)
    state_on, model_on = _channel(z_coord=z, implicit_vmix_e3t_now_divisor=True,
                                  **lf)
    for _ in range(5):
        state_off = model_off.step(state_off, _DT)
        state_on = model_on.step(state_on, _DT)
    dT = float(np.max(np.abs(np.asarray(state_on.T.data)
                             - np.asarray(state_off.T.data))))
    assert dT > 0.0, ("leapfrog ON == OFF: eta_now is NOT threaded at the "
                      "_leapfrog_step call site (fallback reads the AFTER eta)")


def test_e3t_now_divisor_differs_with_eta_tendency():
    """With a NONZERO eta tendency between NOW (eta_now) and AFTER
    (state.eta, what the default dz_cell divisor is built from -- mimicking
    state_corr.eta post-barotropic-solve), the NOW-divisor (this option) and
    the default AFTER-divisor read DIFFERENT thicknesses, so the solved
    T/S/u/v must differ.

    Direction: the implicit solve's diagonal is
    ``dz_cell - (zwi+zws)`` with zwi,zws ~ -p2dt*K/dz_half (a NEGATIVE
    off-diagonal coupling term).  Here eta_now < state.eta (column was
    SHALLOWER at NOW, EXPANDED by the barotropic solve to AFTER) so the
    NOW-divisor dz_half is SMALLER than the default AFTER-divisor
    everywhere -> |zwi|,|zws| LARGER under nemo_kmm -> a MORE dissipative
    (stronger vertical coupling) solve than the default for this scenario.
    This test asserts only that the two differ (a magnitude/sign difference
    is the point); the qualitative diagonal-strength direction is documented
    for this specific eta_now<state.eta construction, matching the existing
    test_active_on_u_centered pattern for the sibling flag."""
    z = _midpoint_zstar()
    state, model = _channel(z_coord=z, implicit_vertical_mixing=True)
    # eta_now UNIFORMLY 20% shallower than state.eta (an expanding column
    # between NOW and AFTER) -- a controlled, nonzero eta tendency.
    eta_now = state.eta.data - 0.2 * jnp.abs(state.eta.data + 10.0)
    out_off = model._apply_implicit_vertical_mixing(state, _DT, None)
    _, model_on = _channel(
        z_coord=z, implicit_vertical_mixing=True,
        implicit_vmix_e3t_now_divisor=True)
    out_on = model_on._apply_implicit_vertical_mixing(
        state, _DT, None, eta_now=eta_now)
    for k in ("T", "S", "u", "v"):
        a = np.asarray(getattr(out_off, k).data)
        b = np.asarray(getattr(out_on, k).data)
        assert np.max(np.abs(a - b)) > 0.0, f"{k} unchanged by e3t_now divisor"
