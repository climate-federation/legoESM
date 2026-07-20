"""Green's-function tests for spectral (Gaussian-grid) tracer transport.

The spectral basis makes the transport operator's Green's function
EXPLICIT, so the tests pin the propagator itself rather than norms:

1. Solid-body rotation about the polar axis (u = Omega a cos(lat), v = 0):
   every spherical harmonic Y_l^m is an EIGENFUNCTION of the advection
   operator with eigenvalue -i m Omega, so the discrete propagator must be
   DIAGONAL (a delta in spectral space stays a delta — no mode leakage) and
   the surviving coefficient must carry the exact phase e^{-i m Omega t}.
2. Spectral hyperdiffusion: the tendency of a unit Y_l^m is EXACTLY
   -nu (l(l+1)/a^2)^order (the diagonal heat-kernel rate), machine-checked
   at the operator level and against exp decay after integration.
3. A band-limited blob advected one full revolution returns to its initial
   state (the classic Williamson-1-style closure; spectral space error is
   pure RK3 time error).

Plus: uniform-tracer preservation under a DIVERGENT wind (advective form),
global-mean (l=0) invariance, vertical-advection wiring, registry
buildability, dispatch hardening, and differentiability.

Run with JAX_ENABLE_X64=1 (spectral = x64 + complex128 policy).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (  # noqa: E402
    SpectralTracerTransportModel,
    TracerTransportSpectralConfig,
    spectral_tracer_grid_values,
    spectral_tracer_state,
    tracer_tendencies_spectral,
)
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import TracerState  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask  # noqa: E402

from legoesm import constants  # noqa: E402

_OMEGA = 2.0 * jnp.pi / (12.0 * 86400.0)   # solid-body rate: one rev / 12 days


def _grid():
    return create_gaussian_grid(n_max=21, dealiasing="linear")


def _solid_body_wind(t, grid, sigma_coord, nlev=1):
    """u = Omega a cos(lat), v = 0 (non-divergent, polar-axis rotation)."""
    del t, sigma_coord
    u = _OMEGA * grid.radius * jnp.cos(grid.lat2d)[:, :, None]
    u = jnp.broadcast_to(u, (grid.n_lat, grid.n_lon, nlev))
    return u, jnp.zeros_like(u), None


def _mode_index(grid, l, m):  # noqa: E741 - l is the standard SH degree symbol
    idx = jnp.where((grid.ls == l) & (grid.ms == m), size=1)[0][0]
    return int(idx)


def _single_mode_state(grid, l, m, nlev=1, n_tracers=1):  # noqa: E741
    q_hat = jnp.zeros((grid.ls.shape[0], nlev, n_tracers), dtype=jnp.complex128)
    q_hat = q_hat.at[_mode_index(grid, l, m), :, :].set(1.0 + 0.0j)
    return TracerState(
        tracers=Field(data=q_hat, name="tracers_hat",
                      dims=("n_sh", "lev", "tracer"), units="1"),
        time=Field(data=jnp.asarray(0.0), name="time", dims=(), units="s"),
    )


# ---------------------------------------------------------------------------
# 1. Solid-body advection: diagonal propagator + exact eigen-phase.
# ---------------------------------------------------------------------------
def test_solid_body_tendency_is_diagonal_eigenvalue():
    """At the OPERATOR level: d(q_hat)/dt of Y_5^3 under solid-body rotation
    is -i m Omega on that single coefficient and ~0 on every other mode
    (the Green's function of advection is diagonal in the SH basis)."""
    grid = _grid()
    l, m = 5, 3  # noqa: E741
    state = _single_mode_state(grid, l, m)
    tend = tracer_tendencies_spectral(
        state, grid, None, _solid_body_wind).tracers.data[:, 0, 0]
    idx = _mode_index(grid, l, m)
    expected = -1j * m * _OMEGA
    got = complex(tend[idx])
    assert abs(got - expected) <= 1e-12 * abs(expected), (
        f"eigenvalue {got} != -i*m*Omega {expected}"
    )
    leak = jnp.abs(tend.at[idx].set(0.0))
    assert float(jnp.max(leak)) <= 1e-13 * abs(expected), (
        f"advection operator leaked {float(jnp.max(leak)):.3e} to other "
        f"modes -> propagator not diagonal"
    )
    # dtime_dt = 1 (stage-time bookkeeping, mirrors the lat-lon sibling).
    t_dot = tracer_tendencies_spectral(
        state, grid, None, _solid_body_wind).time.data
    assert float(t_dot) == 1.0


def test_solid_body_integrated_phase_rotation():
    """Integrated propagator: after time T the Y_5^3 coefficient equals
    e^{-i m Omega T} (RK3 phase/amplitude error only) and no other mode has
    been excited."""
    grid = _grid()
    l, m = 5, 3  # noqa: E741
    model = SpectralTracerTransportModel(grid, None, _solid_body_wind)
    state = _single_mode_state(grid, l, m)
    dt = 1800.0
    n_steps = 96                                # T = 2 days
    for _ in range(n_steps):
        state = model.step(state, dt)
    t_total = n_steps * dt
    idx = _mode_index(grid, l, m)
    got = complex(state.tracers.data[idx, 0, 0])
    expected = np.exp(-1j * m * float(_OMEGA) * t_total)
    # The discrete propagator is R(z)^N with z = -i m Omega dt and R the RK3
    # stability polynomial; |R(z) - e^z| = |z|^4/24 + O(|z|^5), so the total
    # defect is N |z|^4 / 24.  Pinning the error AGAINST that bound checks the
    # propagator to leading order (observed/bound ~ 1.0), not just "small".
    z = m * float(_OMEGA) * dt
    rk3_bound = n_steps * z**4 / 24.0
    err = abs(got - expected)
    assert err < 1.5 * rk3_bound, (
        f"phase error {err:.3e} exceeds 1.5x the RK3 defect {rk3_bound:.3e}"
    )
    leak = jnp.abs(state.tracers.data[:, 0, 0].at[idx].set(0.0))
    assert float(jnp.max(leak)) < 1e-11


def test_high_l_in_band_mode_still_advects():
    """A HIGH-degree mode inside the Orszag band (l=13 <= floor(0.667*21)=14)
    must advect with the exact eigen-phase — a mask cutoff mistakenly applied
    below the documented band (or to the state prematurely) would freeze it
    (codex R1: the band edge is where a masking bug bites)."""
    grid = _grid()
    l, m = 13, 3  # noqa: E741
    state = _single_mode_state(grid, l, m)
    tend = tracer_tendencies_spectral(
        state, grid, None, _solid_body_wind).tracers.data[:, 0, 0]
    idx = _mode_index(grid, l, m)
    expected = -1j * m * _OMEGA
    assert abs(complex(tend[idx]) - expected) <= 1e-11 * abs(expected)


def test_above_band_mode_is_truncated_not_frozen():
    """Band-limit CONTRACT: an initial mode ABOVE the Orszag cutoff
    (l=16 > 14 at T21) is REMOVED by the post-step state truncation — not
    carried frozen at unit amplitude (which would masquerade as transported
    tracer that never moves)."""
    grid = _grid()
    state = _single_mode_state(grid, 16, 3)
    model = SpectralTracerTransportModel(grid, None, _solid_body_wind)
    stepped = model.step(state, 1800.0)
    idx = _mode_index(grid, 16, 3)
    assert abs(complex(stepped.tracers.data[idx, 0, 0])) == 0.0
    # The rest of the spectrum carries only transform round-off from the
    # product terms (~1e-16), NOT the frozen unit amplitude.
    assert float(jnp.max(jnp.abs(stepped.tracers.data))) < 1e-13


# ---------------------------------------------------------------------------
# 2. Hyperdiffusion: exact diagonal heat-kernel rate.
# ---------------------------------------------------------------------------
def test_hyperdiffusion_greens_rate_exact():
    """Zero wind, nu > 0: the tendency of a unit Y_l^m is exactly
    -nu (l(l+1)/a^2)^order on that mode and zero elsewhere."""
    grid = _grid()
    l, m = 7, 4  # noqa: E741
    nu, order = 1.0e15, 2
    cfg = TracerTransportSpectralConfig(hyperdiff_coeff=nu, hyperdiff_order=order)

    def _calm(t, g, sc, nlev=1):
        z = jnp.zeros((g.n_lat, g.n_lon, nlev))
        return z, z, None

    state = _single_mode_state(grid, l, m)
    tend = tracer_tendencies_spectral(
        state, grid, None, _calm, cfg).tracers.data[:, 0, 0]
    idx = _mode_index(grid, l, m)
    rate = -nu * (l * (l + 1) / grid.radius**2) ** order
    got = complex(tend[idx])
    assert abs(got - rate) <= 1e-12 * abs(rate)
    leak = jnp.abs(tend.at[idx].set(0.0))
    assert float(jnp.max(leak)) <= 1e-13 * abs(rate)


def test_hyperdiffusion_integrated_exponential_decay():
    """Integrated heat kernel: |q_hat(T)| = exp(rate * T) within RK3 error."""
    grid = _grid()
    l, m = 7, 4  # noqa: E741
    nu, order = 1.0e15, 2
    cfg = TracerTransportSpectralConfig(hyperdiff_coeff=nu, hyperdiff_order=order)

    def _calm(t, g, sc, nlev=1):
        z = jnp.zeros((g.n_lat, g.n_lon, nlev))
        return z, z, None

    model = SpectralTracerTransportModel(grid, None, _calm, cfg)
    state = _single_mode_state(grid, l, m)
    rate = -nu * (l * (l + 1) / grid.radius**2) ** order
    dt = 900.0
    assert abs(rate) * dt < 0.05                # resolved decay (RK3 accurate)
    n_steps = 48
    for _ in range(n_steps):
        state = model.step(state, dt)
    got = abs(complex(state.tracers.data[_mode_index(grid, l, m), 0, 0]))
    expected = float(np.exp(rate * n_steps * dt))
    # RK3 propagator defect: |R(z) - e^z| = |z|^4/24 per step, z = rate*dt.
    rk3_bound = n_steps * (abs(rate) * dt) ** 4 / 24.0
    assert abs(got - expected) < 1.5 * rk3_bound * expected + 1e-14


# ---------------------------------------------------------------------------
# 3. Band-limited blob: one full revolution returns the initial condition.
# ---------------------------------------------------------------------------
def test_blob_full_revolution_returns_initial():
    grid = _grid()
    lon0, lat0, width = 1.5 * jnp.pi, 0.35, 0.45
    dist2 = (
        (jnp.cos(grid.lat2d) * (jnp.mod(grid.lon2d - lon0 + jnp.pi,
                                        2.0 * jnp.pi) - jnp.pi)) ** 2
        + (grid.lat2d - lat0) ** 2
    )
    blob = jnp.exp(-dist2 / width**2)[:, :, None, None]
    state0 = spectral_tracer_state(grid, blob)
    # Band-limit the INITIAL coefficients with the model's own Orszag mask.
    # The model TRUNCATES the state to this band after every step (see
    # test_above_band_mode_is_truncated_not_frozen), so pre-filtering the IC
    # makes t=0 and t=T live in the same band and the comparison pure
    # transport (otherwise q0 would carry upper-band content the model
    # removes on step 1).
    mask = dealiasing_mask(grid, 0.667)[:, None, None]
    state0 = state0._replace(
        tracers=state0.tracers.replace(data=state0.tracers.data * mask))

    model = SpectralTracerTransportModel(grid, None, _solid_body_wind)
    t_rev = float(2.0 * jnp.pi / _OMEGA)
    n_steps = 1024
    dt = t_rev / n_steps
    state = state0
    for _ in range(n_steps):
        state = model.step(state, dt)

    q0 = spectral_tracer_grid_values(grid, state0)
    q_end = spectral_tracer_grid_values(grid, state)
    rel = float(jnp.linalg.norm(q_end - q0) / jnp.linalg.norm(q0))
    assert rel < 1e-3, f"blob did not return after one revolution: rel={rel:.2e}"


# ---------------------------------------------------------------------------
# 4. Advective-form invariants.
# ---------------------------------------------------------------------------
def test_uniform_tracer_preserved_under_divergent_wind():
    """Advective form: q = const has zero tendency even for div(V) != 0
    (the q*div(V) correction cancels the flux-form source)."""
    grid = _grid()

    def _divergent(t, g, sc, nlev=1):
        u = 10.0 * jnp.sin(g.lon2d)[:, :, None] * jnp.cos(g.lat2d)[:, :, None]
        v = 5.0 * jnp.cos(g.lon2d)[:, :, None] * jnp.cos(g.lat2d)[:, :, None]
        u = jnp.broadcast_to(u, (g.n_lat, g.n_lon, nlev))
        v = jnp.broadcast_to(v, (g.n_lat, g.n_lon, nlev))
        return u, v, None

    ones = jnp.ones((grid.n_lat, grid.n_lon, 1, 1))
    state = spectral_tracer_state(grid, ones)
    tend = tracer_tendencies_spectral(
        state, grid, None, _divergent).tracers.data
    # The two transform paths of the cancellation see identical products, so
    # the residue is transform round-off only.
    assert float(jnp.max(jnp.abs(tend))) < 1e-10


def test_global_mean_invariant_solid_body():
    """The (l=0, m=0) coefficient (global mean) is untouched by advection."""
    grid = _grid()
    blob = (1.0 + 0.3 * jnp.sin(grid.lat2d) * jnp.cos(2.0 * grid.lon2d))
    state = spectral_tracer_state(grid, blob[:, :, None, None])
    mask = dealiasing_mask(grid, 0.667)[:, None, None]
    state = state._replace(
        tracers=state.tracers.replace(data=state.tracers.data * mask))
    idx00 = _mode_index(grid, 0, 0)
    c0 = complex(state.tracers.data[idx00, 0, 0])
    model = SpectralTracerTransportModel(grid, None, _solid_body_wind)
    for _ in range(24):
        state = model.step(state, 1800.0)
    c_end = complex(state.tracers.data[idx00, 0, 0])
    assert abs(c_end - c0) < 1e-11 * abs(c0)


def test_multi_tracer_level_independence():
    """Two tracers x two levels advect independently: tracer 0 = Y_5^3,
    tracer 1 = Y_4^2; each keeps its own eigen-phase with no cross-talk."""
    grid = _grid()
    nlev, ntr = 2, 2
    q_hat = jnp.zeros((grid.ls.shape[0], nlev, ntr), dtype=jnp.complex128)
    q_hat = q_hat.at[_mode_index(grid, 5, 3), :, 0].set(1.0 + 0.0j)
    q_hat = q_hat.at[_mode_index(grid, 4, 2), :, 1].set(1.0 + 0.0j)
    state = TracerState(
        tracers=Field(data=q_hat, name="tracers_hat",
                      dims=("n_sh", "lev", "tracer"), units="1"),
        time=Field(data=jnp.asarray(0.0), name="time", dims=(), units="s"),
    )

    def _wind(t, g, sc):
        return _solid_body_wind(t, g, sc, nlev=nlev)

    model = SpectralTracerTransportModel(grid, None, _wind)
    dt, n_steps = 1800.0, 48
    for _ in range(n_steps):
        state = model.step(state, dt)
    t_total = dt * n_steps
    got0 = complex(state.tracers.data[_mode_index(grid, 5, 3), 1, 0])
    got1 = complex(state.tracers.data[_mode_index(grid, 4, 2), 0, 1])
    # RK3 defect bound per mode (see test_solid_body_integrated_phase_rotation).
    bound_m3 = 1.5 * n_steps * (3 * float(_OMEGA) * dt) ** 4 / 24.0
    bound_m2 = 1.5 * n_steps * (2 * float(_OMEGA) * dt) ** 4 / 24.0
    assert abs(got0 - np.exp(-1j * 3 * float(_OMEGA) * t_total)) < bound_m3
    assert abs(got1 - np.exp(-1j * 2 * float(_OMEGA) * t_total)) < bound_m2
    # No cross-tracer leakage.
    assert abs(complex(state.tracers.data[_mode_index(grid, 4, 2), 0, 0])) < 1e-11
    assert abs(complex(state.tracers.data[_mode_index(grid, 5, 3), 0, 1])) < 1e-11


# ---------------------------------------------------------------------------
# 5. Vertical advection wiring, dispatch, registry, differentiability.
# ---------------------------------------------------------------------------
def test_vertical_advection_wiring_and_guard():
    """sigma_dot without a sigma_coord raises; with one, a vertically-sheared
    tracer under uniform subsidence gets a nonzero, finite tendency."""
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = _grid()
    nlev = 4
    sigma = create_sigma_coordinate(nlev)

    def _subsiding(t, g, sc, nlev=nlev):
        z = jnp.zeros((g.n_lat, g.n_lon, nlev))
        sdot = jnp.full((g.n_lat, g.n_lon, nlev + 1), 1e-4)
        return z, z, sdot

    q = jnp.linspace(1.0, 2.0, nlev)[None, None, :, None]
    q = jnp.broadcast_to(q, (grid.n_lat, grid.n_lon, nlev, 1))
    state = spectral_tracer_state(grid, q)
    with pytest.raises(ValueError, match="sigma_coord"):
        tracer_tendencies_spectral(state, grid, None, _subsiding)
    tend = tracer_tendencies_spectral(
        state, grid, sigma, _subsiding).tracers.data
    assert bool(jnp.all(jnp.isfinite(tend)))
    assert float(jnp.max(jnp.abs(tend))) > 0.0


def test_unknown_integrator_raises():
    grid = _grid()
    model = SpectralTracerTransportModel(
        grid, None, _solid_body_wind,
        TracerTransportSpectralConfig(time_integrator="not_a_scheme"))
    state = _single_mode_state(grid, 5, 3)
    with pytest.raises((ValueError, KeyError)):
        model.step(state, 100.0)


def test_registry_entry_and_buildability():
    from importlib import import_module

    from legoesm.supported_matrix import SUPPORTED_MATRIX

    entries = [
        e for e in SUPPORTED_MATRIX
        if (e.component, e.dynamics, e.grid)
        == ("atmosphere", "tracer_transport", "spectral_gaussian")
    ]
    assert len(entries) == 1, "registry must expose exactly one spectral entry"
    e = entries[0]
    cls = getattr(import_module(e.module), e.class_name)
    assert cls is SpectralTracerTransportModel


def test_differentiable_through_steps():
    grid = _grid()
    model = SpectralTracerTransportModel(grid, None, _solid_body_wind)
    base = _single_mode_state(grid, 5, 3)

    def loss(amp):
        st = base._replace(
            tracers=base.tracers.replace(data=amp * base.tracers.data))
        for _ in range(3):
            st = model.step(st, 1800.0)
        return jnp.sum(jnp.abs(st.tracers.data) ** 2)

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g) and abs(g) > 0.0


def test_radius_is_earth_radius():
    """The default grid radius is the shared Earth radius (no literal)."""
    grid = _grid()
    assert float(grid.radius) == pytest.approx(constants.R_earth)
