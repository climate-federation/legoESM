"""WENO5 water-positivity guard on the plane CRM (codex CRM-dycore review).

WENO5-Z is 5th-order but NOT positivity-preserving: a positive stencil can
reconstruct a negative face value, so it drives the positive-definite CRM
tracers (q_v, q_c, q_r, …) below zero, which downstream microphysics reads as a
spurious source. Two guards fix this:

1. Dispatch guard (``compressible_euler_plane``): when
   ``horizontal_advection_scheme="weno5"`` the TRACER legs fall back to the
   monotone van_leer limiter (θ′ keeps weno5). van_leer's 2-cell halo ⊆ weno5's
   3-cell halo, so it never under-halos.
2. Bridge clip (``microphysics/integration``): the plane microphysics bridge
   clips every tracer READ to ≥ 0, so the physics never sees a residual q < 0
   from any non-positivity-preserving path.

These tests pin both, and pin that the reference (van_leer) path is unchanged.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    _van_leer_advection_x,
    _weno5_advection_x,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup(nx=8, ny=8, nlev=10, dx=500.0, LZ=20000.0):
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             dtype=jnp.float64, coriolis_mode="none")
    hc = create_height_coordinate(nlev, H=LZ)
    tm = make_flat_plane_terrain_metric(grid, hc)
    return grid, hc, tm


def test_weno5_undershoots_positive_front_but_van_leer_does_not():
    """Non-vacuous proof of the defect + the fix: advecting a positive top-hat,
    the raw weno5 leg drives it negative (undershoot), the van_leer leg the
    guard substitutes stays ≥ 0."""
    nx = 24
    phi = np.zeros((1, nx, 1))
    phi[0, 8:16, 0] = 1.0                       # positive-definite top-hat
    phi = jnp.asarray(phi)
    u = jnp.ones((1, nx, 1))
    dx, dt = 1.0, 0.2                           # CFL 0.2

    def _min_over_steps(fn):
        p, m = phi, 0.0
        for _ in range(20):
            p = p + dt * fn(p, u, dx)
            m = min(m, float(jnp.min(p)))
        return m

    weno5_min = _min_over_steps(_weno5_advection_x)
    van_leer_min = _min_over_steps(_van_leer_advection_x)
    assert weno5_min < -1e-11, (
        f"expected weno5 to undershoot a positive front; got {weno5_min:.3e}"
    )
    assert van_leer_min >= -1e-13, (
        f"van_leer must stay positive-definite; got {van_leer_min:.3e}"
    )


def _seed_tracer_front(grid, hc, u_ms):
    """Rest state + a uniform x-wind + a sharp positive q_v front (slot 0)."""
    ny, nx, nlev = grid.ny, grid.nx, hc.theta_ref.shape[0]
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = np.zeros((ny, nx, nlev, 1))
    tracers[:, nx // 4:nx // 2, :, 0] = 1.0e-2      # 10 g/kg block
    u = jnp.full((ny, nx, nlev), u_ms)
    return state._replace(
        u=state.u.replace(data=u),
        tracers=state.tracers.replace(data=jnp.asarray(tracers)),
    )


def test_weno5_config_keeps_tracers_nonnegative():
    """The full plane dycore under ``horizontal_advection_scheme="weno5"`` keeps
    the advected tracer ≥ 0 — the dispatch guard routed the tracer legs onto
    van_leer (cf. the raw-scheme undershoot above)."""
    grid, hc, tm = _setup(nx=16, ny=4, nlev=10, dx=500.0)
    cfg = CompressibleEulerConfig(
        horizontal_advection_scheme="weno5",
        semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
        use_coriolis=False, smagorinsky_cs=0.0, sponge_coeff=0.0,
        hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = _seed_tracer_front(grid, hc, u_ms=20.0)
    for _ in range(30):
        state = model.step(state, dt=3.0)        # advective CFL ≈ 0.12
    q = np.asarray(state.tracers.data)
    assert np.all(np.isfinite(q))
    assert q.min() >= -1e-12, f"weno5 config left q<0: min={q.min():.3e}"


def test_guard_routes_weno5_tracers_to_van_leer_tendency():
    """The guard's exact semantics at the slow-tendency level, non-vacuously: on
    ONE fixed state (a SMOOTH sinusoidal q_v so the schemes genuinely differ —
    at a sharp extremum all limiters revert to first order and coincide), the
    weno5 config's TRACER tendency is bit-identical to the van_leer config's
    (guard → van_leer) yet DIFFERS from the upwind1 config's. The upwind1
    inequality proves the tracer tendency is scheme-sensitive, so the van_leer
    equality is the guard at work — not all schemes trivially coinciding."""
    grid, hc, tm = _setup(nx=16, ny=8, nlev=10, dx=500.0)
    base = dict(
        use_coriolis=False, smagorinsky_cs=0.0, sponge_coeff=0.0,
        hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
    )
    ny, nx, nlev = grid.ny, grid.nx, hc.theta_ref.shape[0]
    x = np.arange(nx)
    q = (1.0e-2 * (1.0 + 0.5 * np.sin(2 * np.pi * x / nx)))[None, :, None]
    q = np.broadcast_to(q, (ny, nx, nlev))[..., None]      # positive, smooth
    state = make_rest_state(grid, hc, dtype=jnp.float64)._replace(
        u=make_rest_state(grid, hc, dtype=jnp.float64).u.replace(
            data=jnp.full((ny, nx, nlev), 20.0)),
        tracers=make_rest_state(
            grid, hc, dtype=jnp.float64).tracers.replace(data=jnp.asarray(q)),
    )

    def _dtracers(scheme):
        cfg = CompressibleEulerConfig(
            horizontal_advection_scheme=scheme, **base)
        return np.asarray(plane_compressible_euler_slow_tendencies(
            state, grid, hc, tm, cfg).dtracers_dt.data)

    d_weno5, d_vanleer, d_upwind1 = (
        _dtracers("weno5"), _dtracers("van_leer"), _dtracers("upwind1"))
    # Guard: weno5 tracers ARE van_leer ⇒ bit-identical.
    np.testing.assert_allclose(d_weno5, d_vanleer, rtol=0, atol=0)
    # Non-vacuous: a different monotone scheme gives a different tracer tendency,
    # so the equality above is the guard, not universal scheme coincidence.
    assert not np.allclose(d_weno5, d_upwind1), (
        "weno5 and upwind1 tracer tendencies coincide — the van_leer equality "
        "would then be vacuous"
    )


def test_plane_microphysics_bridge_clips_negative_tracer_read(monkeypatch):
    """The plane microphysics bridge clips negative tracer READS to ≥0 BEFORE
    the scheme sees them — isolated from Kessler's OWN internal negative guards
    (codex CRM-dycore review). A SPY wrapping the scheme records the q_r it is
    handed: with the bridge clip the spy sees q_r ≥ 0 despite a state with
    q_r<0; with ``clip_positive`` monkeypatched to identity the spy sees the raw
    negative — so the bridge READ-clip (not the scheme internals) is what floors
    it. (The prior version compared Kessler tendencies and would pass even with
    the clip disabled, because Kessler internally guards negative rain.)"""
    from legoesm.atmosphere.physics.microphysics import integration
    grid, hc, tm = _setup(nx=4, ny=4, nlev=10, dx=4000.0)
    ny, nx, nlev = grid.ny, grid.nx, hc.theta_ref.shape[0]
    tr = np.zeros((ny, nx, nlev, 3))
    tr[..., 0] = 1.5e-2                            # q_v near/above saturation
    tr[:, :, -1, 2] = -5.0e-3                      # NEGATIVE rain in the state
    state = make_rest_state(grid, hc, dtype=jnp.float64)._replace(
        tracers=make_rest_state(
            grid, hc, dtype=jnp.float64).tracers.replace(data=jnp.asarray(tr)))

    captured = {}
    real_kessler = integration.kessler_microphysics

    def _spy(T, q_v, hydro, *args, **kwargs):
        captured["qr_min"] = float(jnp.min(hydro.q_r))
        return real_kessler(T, q_v, hydro, *args, **kwargs)

    def _run():
        make_microphysics_physics(
            MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig()),
            model_type="plane", dt=20.0,
        )(state, grid, hc, tm)

    monkeypatch.setattr(integration, "kessler_microphysics", _spy)
    _run()
    assert captured["qr_min"] >= 0.0, (
        f"bridge fed q_r<0 to microphysics: {captured['qr_min']:.3e}")
    # Disable the clip → the spy MUST now see the raw negative (proves the clip,
    # not the scheme, floored it → the test is not vacuous).
    monkeypatch.setattr(integration, "clip_positive", lambda x: x)
    _run()
    assert captured["qr_min"] < 0.0, (
        "clip disabled but microphysics still saw q_r≥0 — the test is not "
        "exercising the bridge clip")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
