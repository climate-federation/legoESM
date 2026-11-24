"""Physics hook of the spectral NH dycore (``step(..., physics_fn=...)``).

Pins the contract used by the matrix DCMIP-2025 TC3 spectral row: the
callback returns SPECTRAL tendencies added at every RK stage.  A callback
returning zeros must reproduce the plain dynamics step; the Kessler callback
(shared grid-space core with the MPAS arm) must change theta' and the water
tracers.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (  # noqa: E402
    SpectralCompressibleEulerModel,
    SpectralNHConfig,
    dcmip25_tc3_init_spectral,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from scripts.matrix.run_atmosphere_test_matrix import (  # noqa: E402
    _make_kessler_spectral_nh_physics_fn,
)

_DT = 3.0


@pytest.fixture(scope="module")
def tc3():
    state, hcoord, tmetric = dcmip25_tc3_init_spectral(
        create_gaussian_grid(21), n_levels=40)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=25, semi_implicit_acoustic=True,
        sponge_width=8000.0, sponge_coeff=0.15, hyperdiff_coeff=0.0,
        small_earth_factor=60.0, rotating=False)
    model = SpectralCompressibleEulerModel(
        create_gaussian_grid(21, radius=constants.R_earth / 60.0),
        hcoord, tmetric, cfg)
    return model, state, hcoord


def _leaves(s):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(s)]


def test_zero_physics_fn_equals_plain_step(tc3):
    model, state, _ = tc3

    def zero_fn(s, grid, hcoord, tmetric):
        return jax.tree_util.tree_map(jnp.zeros_like, s)

    plain = _leaves(model.step(state, _DT))
    with_zero = _leaves(model.step(state, _DT, physics_fn=zero_fn))
    for a, b in zip(plain, with_zero):
        np.testing.assert_array_equal(a, b)


def test_kessler_physics_fn_changes_step(tc3):
    model, state, hcoord = tc3
    phys = _make_kessler_spectral_nh_physics_fn(hcoord, _DT)

    tend = phys(state, model.grid, model.height_coord, model.terrain_metric)
    assert float(jnp.max(jnp.abs(tend.theta_prime_hat.data))) > 0.0
    assert float(jnp.max(jnp.abs(tend.tracers_hat.data))) > 0.0
    for f in ("vor_hat", "div_hat", "w_hat", "rho_prime_hat", "phis_hat"):
        assert float(jnp.max(jnp.abs(getattr(tend, f).data))) == 0.0

    plain = model.step(state, _DT)
    moist = model.step(state, _DT, physics_fn=phys)
    assert np.all(np.isfinite(np.asarray(moist.tracers_hat.data)))
    d_theta = np.max(np.abs(np.asarray(
        moist.theta_prime_hat.data - plain.theta_prime_hat.data)))
    d_q = np.max(np.abs(np.asarray(
        moist.tracers_hat.data - plain.tracers_hat.data)))
    assert d_theta > 0.0 and d_q > 0.0


def test_physics_fn_evaluated_at_every_rk_stage(tc3):
    """The callback runs once per outer SSP-RK3 stage (3 trace-time calls
    in one step), and a ``(tendencies, diagnostics)`` tuple is unwrapped."""
    model, state, _ = tc3
    calls = []

    def counting_zero_fn(s, grid, hcoord, tmetric):
        calls.append(1)
        return jax.tree_util.tree_map(jnp.zeros_like, s), {}

    out = model.step(state, _DT, physics_fn=counting_zero_fn)
    assert len(calls) == 3
    plain = _leaves(model.step(state, _DT))
    for a, b in zip(plain, _leaves(out)):
        np.testing.assert_array_equal(a, b)


def _terrain_metric(model, hcoord):
    """A J != 1, spatially varying terrain metric on the model grid (the
    TC3 case is flat; the adapters must still divide/multiply by J)."""
    from legoesm.grids.vertical import compute_terrain_metric
    g = model.grid
    zs = 1500.0 * (1.0 + jnp.cos(g.lat)[:, None] * jnp.sin(g.lon2d)) / 2.0
    tm = compute_terrain_metric(zs, hcoord)
    assert float(jnp.ptp(tm.jacobian)) > 0.05
    return tm


def test_kessler_adapter_returns_tracer_density_tendency(tc3):
    """tracers_hat is J*rho*q and rho_prime_hat is J*rho': the adapter must
    hand Kessler q = (J rho q)/(J rho) and rho' = (J rho')/J, and return
    J * rho * dq/dt (rho varies 1.2 -> 0.09 kg/m^3 with height, so a missing
    rho is a factor-of-ten error; J != 1 here so a missing J shows too)."""
    from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
    from scripts.matrix.run_atmosphere_test_matrix import (
        _make_kessler_nh_grid_fn,
    )
    model, state, hcoord = tc3
    g = model.grid
    tm = _terrain_metric(model, hcoord)
    J = tm.jacobian
    n_sh, nlev, n_tr = state.tracers_hat.data.shape
    tend = _make_kessler_spectral_nh_physics_fn(hcoord, _DT)(
        state, g, model.height_coord, tm)
    theta_p = sh_synthesis_3d(g, state.theta_prime_hat.data)
    rho_p = sh_synthesis_3d(g, state.rho_prime_hat.data) / J[..., None]
    rho = (hcoord.rho_ref + rho_p)[..., None]
    rq = sh_synthesis_3d(g, state.tracers_hat.data.reshape(n_sh, nlev * n_tr)
                         ).reshape(g.n_lat, g.n_lon, nlev, n_tr) / J[..., None, None]
    _, dq = _make_kessler_nh_grid_fn(hcoord, _DT)(theta_p, rho_p, rq / rho)
    expect = sh_analysis_3d(
        g, (J[..., None, None] * rho * dq).reshape(g.n_lat, g.n_lon, nlev * n_tr)
    ).reshape(n_sh, nlev, n_tr)
    assert float(jnp.max(jnp.abs(expect))) > 0.0
    np.testing.assert_allclose(np.asarray(tend.tracers_hat.data),
                               np.asarray(expect), rtol=0, atol=1e-18)
    assert float(rho.max() / rho.min()) > 5.0


def test_mixing_ratio_physics_adapter_converts_legacy_convention(tc3):
    """Opt-in adapter for physics written for (q, unweighted rho') in and
    (dq/dt, drho'/dt) out: the legacy function must SEE q and rho' and its
    tendencies must come back as d(J rho q)/dt = J (rho dq/dt + q drho'/dt)
    and d(J rho')/dt = J drho'/dt; vor/div/theta' pass through; the tuple
    return and the stateful tag survive."""
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        mixing_ratio_physics_adapter,
    )
    from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
    from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
    from legoesm.timestepping.integration import physics_requires_phys_state
    model, state, hcoord = tc3
    g = model.grid
    tm = _terrain_metric(model, hcoord)
    J = tm.jacobian
    n_sh, nlev, n_tr = state.tracers_hat.data.shape
    # Non-trivial rho' so the q * drho' term and rho != rho_ref matter.
    rho_p_in = 0.01 * hcoord.rho_ref * jnp.cos(g.lon2d)[..., None]
    state = state._replace(rho_prime_hat=state.rho_prime_hat.replace(
        data=sh_analysis_3d(g, J[..., None] * rho_p_in)))
    zs = hcoord.z_full[None, None, :] / 10000.0
    dq_grid = jnp.stack([1e-6 * jnp.sin(g.lon2d)[..., None] * zs
                         * (t + 1) for t in range(n_tr)], axis=-1)
    drho_grid = 1e-5 * jnp.cos(2 * g.lon2d)[..., None] * zs
    seen = {}

    from legoesm.core.field import Field

    def legacy(s, grid, hc, tmetric):
        seen["state"] = s
        tend = jax.tree_util.tree_map(jnp.zeros_like, s)
        # Legacy code builds its OWN output fields with legacy metadata.
        tend = tend._replace(
            theta_prime_hat=tend.theta_prime_hat.replace(
                data=jnp.ones_like(s.theta_prime_hat.data)),
            rho_prime_hat=Field(
                data=sh_analysis_3d(grid, drho_grid), name="rho_prime_hat",
                dims=s.rho_prime_hat.dims, units="kg/m^3/s"),
            tracers_hat=Field(data=sh_analysis_3d(
                grid, dq_grid.reshape(g.n_lat, g.n_lon, nlev * n_tr)
            ).reshape(n_sh, nlev, n_tr), name="tracers_hat",
                dims=s.tracers_hat.dims, units="kg/kg/s"))
        return tend, {"diag": 1}

    legacy._requires_phys_state = True
    wrapped = mixing_ratio_physics_adapter(legacy)
    assert physics_requires_phys_state(wrapped)
    # A legacy-encoded state is refused, not decoded as J-weighted.
    with pytest.raises(ValueError):
        wrapped(state._replace(rho_prime_hat=state.rho_prime_hat.replace(
            name="rho_prime_hat")), g, hcoord, tm)
    # The spectral NH step threads no PhysicsState: a stateful legacy
    # callback must be refused loudly through the adapter too.
    with pytest.raises(NotImplementedError):
        model.step(state, _DT, physics_fn=wrapped)
    tend, diag = wrapped(state, g, hcoord, tm)
    assert diag == {"diag": 1}
    # What the legacy function saw: unweighted rho', q in kg/kg.
    theta_p = sh_synthesis_3d(g, state.theta_prime_hat.data)
    rho_p = sh_synthesis_3d(g, state.rho_prime_hat.data) / J[..., None]
    _, rho = sanitize_theta_rho(hcoord.theta_ref + theta_p, hcoord.rho_ref + rho_p)
    q = (sh_synthesis_3d(g, state.tracers_hat.data.reshape(n_sh, nlev * n_tr)
                         ).reshape(g.n_lat, g.n_lon, nlev, n_tr)
         / J[..., None, None] / rho[..., None])
    v = seen["state"]
    assert v.tracers_hat.units == "kg/kg"
    assert (v.rho_prime_hat.name, v.tracers_hat.name) == (
        "rho_prime_hat", "tracers_hat")                 # legacy metadata in
    np.testing.assert_allclose(
        np.asarray(v.rho_prime_hat.data), np.asarray(sh_analysis_3d(g, rho_p)),
        rtol=0, atol=1e-18)
    np.testing.assert_allclose(
        np.asarray(v.tracers_hat.data),
        np.asarray(sh_analysis_3d(g, q.reshape(g.n_lat, g.n_lon, nlev * n_tr)
                                  ).reshape(n_sh, nlev, n_tr)),
        rtol=0, atol=1e-16)
    # Converted tendencies, against the analytic conversion of the GRID
    # tendencies the legacy function returned.
    drho_s = sh_synthesis_3d(g, sh_analysis_3d(g, drho_grid))
    dq_s = sh_synthesis_3d(g, sh_analysis_3d(
        g, dq_grid.reshape(g.n_lat, g.n_lon, nlev * n_tr))
    ).reshape(g.n_lat, g.n_lon, nlev, n_tr)
    exp_tr = sh_analysis_3d(g, (J[..., None, None] * (
        rho[..., None] * dq_s + q * drho_s[..., None])).reshape(
            g.n_lat, g.n_lon, nlev * n_tr)).reshape(n_sh, nlev, n_tr)
    exp_rho = sh_analysis_3d(g, J[..., None] * drho_s)
    assert float(jnp.max(jnp.abs(exp_tr))) > 0.0
    np.testing.assert_allclose(np.asarray(tend.tracers_hat.data),
                               np.asarray(exp_tr), rtol=0,
                               atol=1e-12 * float(jnp.max(jnp.abs(exp_tr))))
    np.testing.assert_allclose(np.asarray(tend.rho_prime_hat.data),
                               np.asarray(exp_rho), rtol=0,
                               atol=1e-12 * float(jnp.max(jnp.abs(exp_rho))))
    assert tend.tracers_hat.units == "kg/m^3"
    assert (tend.rho_prime_hat.name, tend.tracers_hat.name) == (
        state.rho_prime_hat.name, state.tracers_hat.name)  # J-weighted out
    np.testing.assert_array_equal(np.asarray(tend.theta_prime_hat.data),
                                  np.ones_like(state.theta_prime_hat.data))
    # The q * drho' term is not negligible here (it would hide a dropped term).
    q_term = sh_analysis_3d(g, (J[..., None, None] * q * drho_s[..., None]).reshape(
        g.n_lat, g.n_lon, nlev * n_tr))
    assert float(jnp.max(jnp.abs(q_term))) > 1e-3 * float(jnp.max(jnp.abs(exp_tr)))
