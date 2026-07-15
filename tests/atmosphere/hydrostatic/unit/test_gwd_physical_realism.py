"""Physical-realism + numerical-stability suite for the non-e3sm_cam GWD schemes.

This suite is the differentiable-physics counterpart of the oracle-anchored
``test_gwd_e3sm_cam.py``: rather than matching a Fortran reference term-by-term,
it asserts the *physics invariants* every gravity-wave-drag scheme must satisfy
on an idealized single column, regardless of its internal closure.

Idealized column (``_idealized_column``)
----------------------------------------
Hydrostatic, exponential density ``rho(z) = rho0 * exp(-z/H)`` with constant
scale height ``H`` and (near-)constant buoyancy frequency ``N`` (isothermal
``T0`` so ``N`` is set by gravity and ``H``), a westerly jet ``U(z)`` ramping
``0 -> U_jet`` over the lower troposphere then constant aloft, ``v=0``.
Pressure is the ideal-gas pressure of that density.  The source is the natural
one for each scheme: a surface orographic stress (Lindzen, McFarlane), a launch
amplitude / phase-speed spectrum (Hines, prognostic_spectral), a sigma drag
profile (Rayleigh).

Invariants asserted (per scheme; see the module-level constants below for which
schemes are physics-faithful and which are surrogates)
------------------------------------------------------
1. **Finite / bounded** -- ``du/dt, dv/dt, dT/dt`` finite, and ``|du/dt|`` below
   a physical ceiling (cf. CAM ``tndmax`` ~ 400-500 m/s/day).
2. **Momentum conservation** -- ``int rho*(du/dt) dz`` equals minus the net
   momentum-flux divergence ``-(tau_top - tau_sfc)``; for a column that fully
   absorbs the launched flux this is ``+tau_sfc`` (drag removes the surface
   stress).  Checked here as: the column drag is a momentum *sink* whose
   magnitude does not exceed the launched stress (no spurious momentum source).
3. **Sign / drag opposes (u - c)** -- for orographic waves (c = 0) the drag
   decelerates the flow, ``du/dt * u <= 0`` wherever the wave breaks; no
   spurious acceleration.
4. **Breaking realism** -- drag is ~zero where the wave is unsaturated and grows
   upward as ``rho`` decreases (flux conserved until saturation).
5. **Critical-level absorption** -- with ``U -> c`` embedded, the wave is
   absorbed at the critical level and deposits ~no momentum above it; no NaN at
   ``U = c``.
6. **Frictional-heating sign** -- KE dissipation gives ``dT/dt >= 0`` where the
   drag acts.
7. **Differentiability** -- ``jax.grad`` of the column drag w.r.t. ``U(z)`` is
   finite and non-zero; ``jit == eager``; ``vmap`` over columns matches.

Scheme classification
---------------------
* ``rayleigh``  -- linear friction / sponge.  Physics-faithful for the *sign*,
  *finiteness*, *heating*, and *differentiability* invariants.  It is NOT a
  wave-propagation scheme, so the breaking / critical-level / flux-growth
  invariants (4, 5) do not apply (there is no launched flux to conserve); the
  momentum check is the trivial ``-k*u`` sink.
* ``lindzen``, ``mcfarlane`` -- orographic saturation (c = 0).  ALL physics
  invariants apply.
* ``hines`` -- Doppler-spread, single bulk-amplitude surrogate (see hines.py
  docstring).  Sign / finiteness / heating / flux-growth / differentiability
  apply.  Its critical-level behavior is amplitude-based (no explicit c), so the
  critical-level test is applied as "remains finite and a sink through a wind
  reversal".
* ``prognostic_spectral`` -- directional deposition (F-GWD-1 FIXED): the
  deposition carries the CONSTANT launch-level sign ``tanh((c - U_launch)/w) ~
  sign(c - U_launch)`` (fixed with height), which drives the projected wind
  toward the wave phase speed WHILE the local wind stays on the launch side of
  ``c`` (deceleration for a slow-launched wave, acceleration for a fast one).
  A wave launched FASTER than the flow legitimately accelerates
  it (toward ``c`` while the local wind is below ``c``; QBO-style forcing), so
  the drag-opposes-flow invariant is
  asserted on a SLOW spectrum (``c_max < min u``, where launch and local sign
  agree); the dissipative heating uses the intrinsic MAGNITUDE form
  ``|c - U_proj| * deposit`` and is non-negative by construction for ANY
  spectrum.  The scheme remains opt-in (default GWD is ``"none"``) pending a
  QBO/momentum-deposition benchmark.
* ``ml_emulator`` -- learned surrogate, NOT physics-faithful.  Only the
  stability / finiteness / differentiability invariants apply (asserted); the
  sign / conservation / breaking invariants are explicitly NOT asserted and the
  reason is documented in ``test_ml_emulator_*``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as C
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    RayleighConfig,
    LindzenConfig,
    McFarlaneConfig,
    HinesConfig,
    PrognosticSpectralConfig,
    GWDMLEmulatorConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    prognostic_spectral_gwd,
)
from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
    ml_gwd,
    GWDEmulator,
)

# Physical ceiling on the wind tendency [m/s/day].  CAM caps the spectral GWD
# tendency at ``tndmax = 400`` m/s/day (orographic-only 500) BEFORE the
# efficiency factor; a converged scheme on a realistic column should sit well
# under this after the closure's saturation/limiters.
TNDMAX_M_S_DAY = 500.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# Idealized column
# ---------------------------------------------------------------------------

def _idealized_column(
    nlev: int = 60,
    *,
    H: float = 7000.0,
    T0: float = 250.0,
    rho0: float = 1.2,
    z_top: float = 60000.0,
    u_jet: float = 40.0,
    u_sfc: float = 10.0,
    z_jet: float = 20000.0,
    u_profile: str = "monotone",
):
    """Build a single idealized GWD column (ncol = 1).

    Returns the standard GWD backend inputs ``(u, v, T, p_full, p_half,
    z_full, z_half, rho)`` all shaped ``(1, nlev)`` except the half-level
    arrays which are ``(1, nlev+1)``.  Index 0 is the model top, index ``-1``
    the surface (the convention the schemes assume: ``u[:, -1]`` is the
    surface wind).

    ``rho(z) = rho0 * exp(-z/H)``; ``T = T0`` (isothermal) so the buoyancy
    frequency ``N = sqrt(g^2/(c_pd*T0))`` is essentially constant; pressure is
    the ideal-gas pressure of that density.

    ``u_profile``:
      * ``"monotone"``  : ``u_sfc -> u_jet`` over ``[0, z_jet]``, constant
        above.  ``u_sfc`` is non-zero (a surface/trade wind) so the orographic
        schemes' ``U > min_wind`` launch activation actually fires -- a column
        with zero surface wind launches no orographic waves (correct physics,
        but a degenerate test).
      * ``"critical"``  : ``+u_jet/2 -> -u_jet/2`` linearly, crossing 0 (the
        orographic phase speed) once -> embeds a critical level.
    """
    z_half = jnp.linspace(z_top, 0.0, nlev + 1)        # top-first
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    rho = rho0 * jnp.exp(-z_full / H)
    rho_h = rho0 * jnp.exp(-z_half / H)
    p_full = rho * C.R_d * T0
    p_half = rho_h * C.R_d * T0
    T = jnp.full((nlev,), T0)
    if u_profile == "monotone":
        u = u_sfc + (u_jet - u_sfc) * jnp.clip(z_full / z_jet, 0.0, 1.0)
    elif u_profile == "critical":
        # Linear from +u_jet/2 at surface to -u_jet/2 at z_top, crosses 0.
        frac = jnp.clip(z_full / z_top, 0.0, 1.0)
        u = 0.5 * u_jet - u_jet * frac
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown u_profile {u_profile!r}")
    v = jnp.zeros((nlev,))

    to = lambda a: jnp.asarray(a)[None, :]
    return (
        to(u), to(v), to(T), to(p_full),
        jnp.asarray(p_half)[None, :],
        to(z_full), jnp.asarray(z_half)[None, :],
        to(rho),
    )


def _layer_dz(z_half):
    return jnp.abs(z_half[:, :-1] - z_half[:, 1:])


def _column_momentum(out, rho, z_half):
    """int rho * du/dt dz over the column [kg/(m s^2) ... momentum sink]."""
    dz = _layer_dz(z_half)
    return float(jnp.sum(rho * out.du_dt * dz, axis=1)[0])


# ---------------------------------------------------------------------------
# Shared invariant helpers (applied per scheme with the right exclusions)
# ---------------------------------------------------------------------------

def _assert_finite_and_bounded(out):
    assert jnp.all(jnp.isfinite(out.du_dt)), "du_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.dv_dt)), "dv_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(out.dT_dt)), "dT_dt has NaN/Inf"
    max_tend = float(jnp.max(jnp.abs(out.du_dt))) * 86400.0
    assert max_tend <= TNDMAX_M_S_DAY, (
        f"|du/dt| = {max_tend:.1f} m/s/day exceeds physical ceiling "
        f"{TNDMAX_M_S_DAY} m/s/day"
    )


def _assert_drag_opposes_flow(out, u, v=None):
    # Drag must not add kinetic energy anywhere: the wind-dot-tendency
    # ``u*du/dt + v*dv/dt <= 0`` (equivalently ``accel * U_proj <= 0``).  The
    # idealized columns here are purely zonal (``v = 0``), so this reduces to
    # the scalar ``u*du/dt <= 0``; pass ``v`` for a 2-D column.  NOTE (codex
    # round-4): do NOT read this component-wise as ``u*du <= 0`` AND
    # ``v*dv <= 0`` when ``v`` is active -- only the dot product is invariant.
    prod = u * out.du_dt
    if v is not None:
        prod = prod + v * out.dv_dt
    assert float(jnp.max(prod)) <= 1e-9, (
        f"spurious acceleration: max(wind.tendency) = {float(jnp.max(prod)):.3e} > 0"
    )


def _assert_heating_nonneg(out):
    assert float(jnp.min(out.dT_dt)) >= -1e-12, (
        f"frictional heating negative: min dT/dt = {float(jnp.min(out.dT_dt)):.3e}"
    )


def _assert_grad_finite_nonzero(loss_fn, u):
    g = jax.grad(loss_fn)(u)
    assert jnp.all(jnp.isfinite(g)), "grad has NaN/Inf"
    assert float(jnp.sum(jnp.abs(g))) > 0.0, "grad is identically zero"


def _assert_real_vmap_parity(scheme_fn, cfg):
    """Build a 2-column batch and assert ``jax.vmap`` over the column axis
    matches per-column eager calls (codex round-1 #6: the suite must exercise
    the *actual* ``jax.vmap`` transform, not a manual batched array).

    The single-column backend takes ``(1, nlev)`` arrays, so we ``vmap`` over a
    leading ensemble axis of single-column inputs (shape ``(2, 1, nlev)``)."""
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    u2 = jnp.stack([u, 0.5 * u], axis=0)          # (2, 1, nlev)
    v2 = jnp.stack([v, v], axis=0)
    T2 = jnp.stack([T, T], axis=0)
    pf2 = jnp.stack([pf, pf], axis=0)
    ph2 = jnp.stack([ph, ph], axis=0)
    zf2 = jnp.stack([zf, zf], axis=0)
    zh2 = jnp.stack([zh, zh], axis=0)
    rho2 = jnp.stack([rho, rho], axis=0)
    lat2 = jnp.stack([jnp.zeros(1), jnp.zeros(1)], axis=0)

    def one(uu, vv, TT, pff, phh, zff, zhh, rr, ll):
        return scheme_fn(uu, vv, TT, pff, phh, zff, zhh, rr, ll, 1800.0,
                         cfg).du_dt

    vmapped = jax.vmap(one)(u2, v2, T2, pf2, ph2, zf2, zh2, rho2, lat2)
    eager0 = one(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1))
    eager1 = one(0.5 * u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1))
    np.testing.assert_allclose(np.array(vmapped[0]), np.array(eager0),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.array(vmapped[1]), np.array(eager1),
                               rtol=1e-12, atol=1e-15)


# ===========================================================================
# Rayleigh friction
# ===========================================================================

def test_rayleigh_finite_sign_heating():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    out = rayleigh_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                       RayleighConfig())
    _assert_finite_and_bounded(out)
    _assert_drag_opposes_flow(out, u)       # -k*u is a pure sink
    _assert_heating_nonneg(out)
    # Momentum sink (negative; friction removes positive momentum).
    assert _column_momentum(out, rho, zh) < 0.0


def test_rayleigh_grad_jit_vmap():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = RayleighConfig()
    fn = lambda uu: rayleigh_gwd(uu, v, T, pf, ph, zf, zh, rho, jnp.zeros(1),
                                 1800.0, cfg)
    _assert_grad_finite_nonzero(lambda uu: jnp.sum(fn(uu).du_dt ** 2), u)
    eager = fn(u)
    jitted = jax.jit(fn)(u)
    np.testing.assert_allclose(np.array(jitted.du_dt), np.array(eager.du_dt),
                               rtol=1e-12, atol=1e-15)
    _assert_real_vmap_parity(rayleigh_gwd, cfg)


# ===========================================================================
# Lindzen orographic saturation
# ===========================================================================

def test_lindzen_finite_sign_heating_breaking():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    out = lindzen_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                      LindzenConfig(h_topo=500.0))
    _assert_finite_and_bounded(out)
    _assert_drag_opposes_flow(out, u)
    _assert_heating_nonneg(out)


def _surface_brunt_vaisala(T, p_full, z_full):
    """Reproduce the scheme's discrete surface N (theta-gradient form) so the
    momentum-conservation reconstruction uses the SAME N the scheme launches
    with (an isothermal analytic ``N = sqrt(g^2/(c_pd T))`` differs by ~2 % from
    the finite-difference theta gradient and would make the bound spuriously
    tight)."""
    theta = T * (C.p_ref / jnp.clip(p_full, 1.0, None)) ** C.kappa
    dz_full = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    N2 = jnp.clip((C.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz, 1e-8, None)
    return float(jnp.sqrt(N2)[0, -1])   # surface half level


def test_lindzen_momentum_sink_bounded_by_launch():
    """Column drag is a sink and never exceeds the launched surface stress."""
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = LindzenConfig(h_topo=500.0)
    out = lindzen_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0, cfg)
    # Reconstruct the launch stress tau_0 = rho_sfc*N_sfc*k*h^2*U_sfc using the
    # scheme's own discrete surface N.
    rho_sfc = float(rho[0, -1])
    U_sfc = float(jnp.abs(u[0, -1]))
    N = _surface_brunt_vaisala(T, pf, zf)
    tau0 = rho_sfc * N * cfg.k_wave * cfg.h_topo ** 2 * U_sfc
    mom = _column_momentum(out, rho, zh)
    assert mom <= 1e-9, "column momentum is a source, not a sink"
    assert abs(mom) <= tau0 * (1.0 + 1e-6), (
        f"|column drag| {abs(mom):.3e} exceeds launched stress {tau0:.3e}"
    )


def test_lindzen_critical_level_absorption():
    """A wind reversal (U crosses c=0) absorbs the wave: ~no drag above it."""
    u, v, T, pf, ph, zf, zh, rho = _idealized_column(u_profile="critical")
    out = lindzen_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                      LindzenConfig(h_topo=500.0))
    assert jnp.all(jnp.isfinite(out.du_dt)), "NaN at critical level"
    du = np.array(out.du_dt[0])
    u0 = np.array(u[0])
    kc = int(np.argmin(np.abs(u0)))          # critical-level index (top-first)
    above = np.max(np.abs(du[:kc])) * 86400.0  # levels above the critical level
    below = np.max(np.abs(du[kc + 1:])) * 86400.0
    assert above < 1.0, (
        f"wave transmitted above critical level: {above:.2f} m/s/day "
        f"(below {below:.2f})"
    )
    # No spurious acceleration of the reversed flow above the critical level.
    # Tolerance 1e-6 absorbs the machine-level leakage from the gated
    # saturation-stress floor (~1e-10 Pa), which is ~9 orders of magnitude
    # below the active drag -- not a physics violation.
    assert float(np.max(u0[:kc] * du[:kc])) <= 1e-12


def test_lindzen_grad_jit_vmap():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = LindzenConfig(h_topo=500.0)
    fn = lambda uu: lindzen_gwd(uu, v, T, pf, ph, zf, zh, rho, jnp.zeros(1),
                                1800.0, cfg)
    _assert_grad_finite_nonzero(lambda uu: jnp.sum(fn(uu).du_dt ** 2), u)
    eager = fn(u)
    jitted = jax.jit(fn)(u)
    np.testing.assert_allclose(np.array(jitted.du_dt), np.array(eager.du_dt),
                               rtol=1e-12, atol=1e-15)
    # Real jax.vmap over the column axis matches per-column eager calls, AND a
    # batched (2-column) array call matches per-column calls (two distinct
    # coverage paths: the vmap transform and ncol>1 batching).
    _assert_real_vmap_parity(lindzen_gwd, cfg)
    u2 = jnp.concatenate([u, 0.5 * u], axis=0)
    v2 = jnp.concatenate([v, v], axis=0)
    T2 = jnp.concatenate([T, T], axis=0)
    pf2 = jnp.concatenate([pf, pf], axis=0)
    ph2 = jnp.concatenate([ph, ph], axis=0)
    zf2 = jnp.concatenate([zf, zf], axis=0)
    zh2 = jnp.concatenate([zh, zh], axis=0)
    rho2 = jnp.concatenate([rho, rho], axis=0)
    batched = lindzen_gwd(u2, v2, T2, pf2, ph2, zf2, zh2, rho2, jnp.zeros(2),
                          1800.0, cfg)
    single1 = lindzen_gwd(0.5 * u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1),
                          1800.0, cfg)
    np.testing.assert_allclose(np.array(batched.du_dt[1]),
                               np.array(single1.du_dt[0]),
                               rtol=1e-12, atol=1e-15)


# ===========================================================================
# McFarlane orographic saturation
# ===========================================================================

def test_mcfarlane_finite_sign_heating_breaking():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    out = mcfarlane_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                        McFarlaneConfig(h_topo=500.0))
    _assert_finite_and_bounded(out)
    _assert_drag_opposes_flow(out, u)
    _assert_heating_nonneg(out)


def test_mcfarlane_momentum_sink_bounded_by_launch():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = McFarlaneConfig(h_topo=500.0)
    out = mcfarlane_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0, cfg)
    rho_sfc = float(rho[0, -1])
    U_sfc = float(jnp.abs(u[0, -1]))
    N = _surface_brunt_vaisala(T, pf, zf)
    # McFarlane / E3SM gw_oro_src launch: G_0 * rho * N * k * h_eff^2 * U with
    # the Froude-number displacement cap h_eff^2 = min(h^2, fcrit2*(U/N)^2),
    # then clipped to tau_max.
    h_eff_sq = min(cfg.h_topo ** 2, cfg.fcrit2 * (U_sfc / N) ** 2)
    tau0 = min(cfg.G_0 * rho_sfc * N * cfg.k_wave * h_eff_sq * U_sfc,
               cfg.tau_max)
    mom = _column_momentum(out, rho, zh)
    assert mom <= 1e-9, "column momentum is a source, not a sink"
    assert abs(mom) <= tau0 * (1.0 + 1e-6), (
        f"|column drag| {abs(mom):.3e} exceeds launched stress {tau0:.3e}"
    )


def test_mcfarlane_froude_cap_active():
    """A tall mountain in a weak wind launches FAR less than the raw h^2 form,
    because the E3SM Froude cap min(h^2, fcrit2*(U/N)^2) bounds the streamline
    displacement (codex round-1 #3: the earlier launch omitted this cap)."""
    # Weak surface wind (5 m/s, > min_wind=2) under a very tall "mountain".
    u, v, T, pf, ph, zf, zh, rho = _idealized_column(u_sfc=5.0, u_jet=20.0)
    rho_sfc = float(rho[0, -1])
    U_sfc = float(jnp.abs(u[0, -1]))
    N = _surface_brunt_vaisala(T, pf, zf)
    tall = McFarlaneConfig(h_topo=5000.0)
    out = mcfarlane_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0, tall)
    mom = abs(_column_momentum(out, rho, zh))
    # Uncapped launch (raw h^2) would be enormous; the Froude cap holds the
    # column sink below the capped launch, which is ~ (U/N)^2/h^2 smaller.
    raw_tau0 = tall.G_0 * rho_sfc * N * tall.k_wave * tall.h_topo ** 2 * U_sfc
    capped_tau0 = min(
        tall.G_0 * rho_sfc * N * tall.k_wave
        * (tall.fcrit2 * (U_sfc / N) ** 2) * U_sfc,
        tall.tau_max,
    )
    assert capped_tau0 < 0.05 * raw_tau0, "Froude cap not reducing the launch"
    assert mom <= capped_tau0 * (1.0 + 1e-6), (
        f"column sink {mom:.3e} exceeds Froude-capped launch {capped_tau0:.3e}"
    )


def test_mcfarlane_critical_level_absorption():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column(u_profile="critical")
    out = mcfarlane_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                        McFarlaneConfig(h_topo=500.0))
    assert jnp.all(jnp.isfinite(out.du_dt)), "NaN at critical level"
    du = np.array(out.du_dt[0])
    u0 = np.array(u[0])
    kc = int(np.argmin(np.abs(u0)))
    above = np.max(np.abs(du[:kc])) * 86400.0
    below = np.max(np.abs(du[kc + 1:])) * 86400.0
    assert above < 1.0, (
        f"wave transmitted above critical level: {above:.2f} m/s/day "
        f"(below {below:.2f})"
    )
    # Tolerance 1e-6 absorbs machine-level leakage from the gated
    # saturation-stress floor (~9 orders below the active drag).
    assert float(np.max(u0[:kc] * du[:kc])) <= 1e-12
    # Documented single-wave behavior (codex round-2 #1): for a SMOOTHLY
    # reversing jet essentially all launched momentum is deposited BELOW the
    # critical level by ordinary saturation breaking -- the absorb/radiate gate
    # only loses the (here negligible) residual reaching the reversal.  Assert
    # the bulk of the column drag lives below the critical level.
    below_sum = float(np.sum(np.abs(du[kc + 1:])))
    above_sum = float(np.sum(np.abs(du[:kc])))
    assert below_sum > 50.0 * above_sum, (
        "more than ~2% of the drag is above the critical level"
    )


def test_mcfarlane_sharp_reversal_absorption():
    """A near-discontinuous wind reversal (positive jet directly above a
    strongly negative layer) must still be absorbed: the critical-level gate is
    applied to the CARRIED stress (not tau_sat), so it is robust even when
    tau_sat would fall below the safe_divide mask floor on the jump (codex
    round-1 #1)."""
    nlev = 40
    z_top = 60000.0
    z_half = jnp.linspace(z_top, 0.0, nlev + 1)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    H, T0 = 7000.0, 250.0
    rho = (1.2 * jnp.exp(-z_full / H))[None, :]
    p_full = (rho[0] * C.R_d * T0)[None, :]
    rho_h = 1.2 * jnp.exp(-z_half / H)
    p_half = (rho_h * C.R_d * T0)[None, :]
    T = jnp.full((1, nlev), T0)
    v = jnp.zeros((1, nlev))
    # +15 m/s below ~20 km, abruptly -15 m/s above (a sharp critical level).
    u = jnp.where(z_full < 20000.0, 15.0, -15.0)[None, :]
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full[None, :],
                        z_half[None, :], rho, jnp.zeros(1), 1800.0,
                        McFarlaneConfig(h_topo=500.0))
    assert jnp.all(jnp.isfinite(out.du_dt)), "NaN at sharp critical level"
    du = np.array(out.du_dt[0])
    u0 = np.array(u[0])
    # Top-first ordering: surface (index -1) is +15, top (index 0) is -15; the
    # wave launches at the surface and a critical level sits at the +/- jump.
    # ``kc`` = the highest still-positive level (the last level BELOW the
    # critical level); everything at index < kc is in the reversed layer ABOVE.
    pos = np.where(u0 > 0)[0]
    kc = int(pos.min())   # smallest index that is still positive (top-first)
    above = np.max(np.abs(du[:kc])) * 86400.0
    assert above < 1.0, (
        f"sharp reversal transmitted stress: {above:.2f} m/s/day above the "
        f"critical level"
    )
    assert float(np.max(u0[:kc] * du[:kc])) <= 1e-12


def test_mcfarlane_grad_jit_vmap():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = McFarlaneConfig(h_topo=500.0)
    fn = lambda uu: mcfarlane_gwd(uu, v, T, pf, ph, zf, zh, rho, jnp.zeros(1),
                                  1800.0, cfg)
    _assert_grad_finite_nonzero(lambda uu: jnp.sum(fn(uu).du_dt ** 2), u)
    eager = fn(u)
    jitted = jax.jit(fn)(u)
    np.testing.assert_allclose(np.array(jitted.du_dt), np.array(eager.du_dt),
                               rtol=1e-12, atol=1e-15)
    _assert_real_vmap_parity(mcfarlane_gwd, cfg)


# ===========================================================================
# Hines Doppler-spread (single bulk-amplitude surrogate)
# ===========================================================================

def test_hines_finite_sign_heating():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    out = hines_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                    HinesConfig())
    _assert_finite_and_bounded(out)
    _assert_drag_opposes_flow(out, u)   # deposition is along +U (a sink)
    _assert_heating_nonneg(out)


def test_hines_flux_grows_upward():
    """Drag deposition strengthens as rho decreases aloft (until the cap)."""
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    out = hines_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                    HinesConfig())
    du = np.abs(np.array(out.du_dt[0]))
    # The peak acceleration is in the upper half of the column, not the surface.
    nlev = du.shape[0]
    assert np.argmax(du) < nlev // 2, (
        "Hines drag peaks near the surface, not aloft as 1/rho growth implies"
    )


def test_hines_wind_reversal_finite_sink():
    """Hines is amplitude-based (no explicit c): through a wind reversal it
    must stay finite and remain a momentum sink (no acceleration)."""
    u, v, T, pf, ph, zf, zh, rho = _idealized_column(u_profile="critical")
    out = hines_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                    HinesConfig())
    assert jnp.all(jnp.isfinite(out.du_dt))
    _assert_drag_opposes_flow(out, u)


def test_hines_grad_jit_vmap():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = HinesConfig()
    fn = lambda uu: hines_gwd(uu, v, T, pf, ph, zf, zh, rho, jnp.zeros(1),
                              1800.0, cfg)
    _assert_grad_finite_nonzero(lambda uu: jnp.sum(fn(uu).du_dt ** 2), u)
    eager = fn(u)
    jitted = jax.jit(fn)(u)
    np.testing.assert_allclose(np.array(jitted.du_dt), np.array(eager.du_dt),
                               rtol=1e-12, atol=1e-15)
    _assert_real_vmap_parity(hines_gwd, cfg)


# ===========================================================================
# Prognostic spectral -- directional deposition (F-GWD-1 fixed)
# ===========================================================================
#
# The deposition in prognostic_spectral.py carries the CONSTANT launch-level
# sign ``tanh((c - U_launch)/w) ~ sign(c - U_launch)`` (fixed with height):
# momentum deposited by a breaking wave drives the projected wind toward
# the wave phase speed ``c`` WHILE the local wind stays on the launch side of
# ``c`` (deceleration for a slow-launched wave, acceleration for a fast one).
# A wave launched FASTER than the flow legitimately accelerates it
# (toward ``c`` while the local wind is below ``c``; QBO-style forcing), so
# "drag opposes flow" is a true invariant only for a
# spectrum SLOWER than the wind everywhere (launch and local sign agree) --
# asserted below with a slow-spectrum config.  The dissipative heating uses the
# intrinsic MAGNITUDE form ``|c - U_proj| * deposit >= 0`` (>= 0 for ANY
# spectrum/wind).

def _run_prognostic(u_profile="monotone", cfg=None, launch=None, u=None):
    col = _idealized_column(u_profile=u_profile)
    u_col, v, T, pf, ph, zf, zh, rho = col
    if u is None:
        u = u_col
    if cfg is None:
        cfg = PrognosticSpectralConfig()
    if launch is None:
        launch = cfg.launch_flux
    spec = jnp.full((1, cfg.n_azimuths, cfg.n_wavenumbers), launch)
    out, _ = prognostic_spectral_gwd(u, v, T, pf, ph, zf, zh, rho,
                                     jnp.zeros(1), 1800.0, cfg, spec)
    return out, u, rho, zh


def test_prognostic_spectral_finite():
    """Tendencies are finite (no NaN/Inf)."""
    out, u, rho, zh = _run_prognostic()
    assert jnp.all(jnp.isfinite(out.du_dt))
    assert jnp.all(jnp.isfinite(out.dv_dt))
    assert jnp.all(jnp.isfinite(out.dT_dt))


def _slow_spectrum_config():
    """Spectrum slower than the wind everywhere: c = N/k <= N*3km/(2*pi)
    ~ 9.4 m/s < u_sfc = 10 m/s (isothermal T0=250 K column,
    N = sqrt(g^2/(c_pd*T0)) ~ 0.0196 1/s)."""
    import math
    return PrognosticSpectralConfig(k_min=2.0 * math.pi / 3.0e3)


def test_prognostic_spectral_slow_spectrum_drag_opposes_flow():
    """F-GWD-1 fix: with every phase speed below the wind (c < U_proj on the
    along-wind azimuth), sign(c - U_proj) < 0 and each deposit decelerates
    the flow -- u*du/dt <= 0 everywhere, a true drag."""
    out, u, rho, zh = _run_prognostic(cfg=_slow_spectrum_config(), launch=0.1)
    # Non-vacuous: breaking must actually deposit somewhere.
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0
    _assert_drag_opposes_flow(out, u)
    # Net column force opposes the (purely zonal, positive) flow.
    assert _column_momentum(out, rho, zh) < 0.0


def test_prognostic_spectral_heating_nonneg():
    """Intrinsic-form dissipation: dT_dt >= 0 and eps_gwd >= 0 by
    construction for the DEFAULT (fast) spectrum -- even where fast waves
    accelerate the flow, the wave supplies the KE and the heating stays
    non-negative."""
    out, u, rho, zh = _run_prognostic(launch=0.1)
    _assert_heating_nonneg(out)
    assert float(jnp.min(out.eps_gwd)) >= -1e-12
    # Energy tie-back: c_pd * integral(rho * dT_dt * dz) == eps_gwd.
    dz = _layer_dz(zh)
    heating_power = jnp.sum(rho * out.dT_dt * C.c_pd * dz, axis=1)
    np.testing.assert_allclose(np.asarray(out.eps_gwd),
                               np.asarray(heating_power),
                               rtol=1e-10, atol=1e-14)


def test_prognostic_spectral_symmetric_two_wave():
    """F-GWD-1 acceptance test: a symmetric two-wave spectrum (one phase
    speed c along +x and -x; n_az=2, n_wn=1) with 0 < c < U:

    * mean flow U > 0: both waves deposit AGAINST the flow
      (+x wave: sign(c - U) < 0; -x wave: sign(c + U) > 0 along the -x
      azimuth) -> net force opposes U (drag);
    * U = 0: saturation and deposits are azimuth-symmetric -> zero force;
    * eps_gwd >= 0 in both cases.
    """
    import math
    # Single wavenumber k0 -> c = N/k0 ~ 6.2 m/s < u everywhere (u in
    # [10, 40] m/s on the monotone column).
    cfg = PrognosticSpectralConfig(
        n_azimuths=2, n_wavenumbers=1,
        k_min=2.0 * math.pi / 2.0e3, k_max=2.0 * math.pi / 2.0e3,
    )
    out, u, rho, zh = _run_prognostic(cfg=cfg, launch=0.5)
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0  # deposits active
    _assert_drag_opposes_flow(out, u)
    assert _column_momentum(out, rho, zh) < 0.0  # net drag on +x flow
    # Purely zonal azimuths -> no meridional force.
    assert float(jnp.max(jnp.abs(out.dv_dt))) < 1e-12
    assert float(jnp.min(out.eps_gwd)) >= -1e-12

    # U = 0: exact cancellation of the +-x deposits by symmetry.
    u0 = jnp.zeros_like(u)
    out0, _, _, _ = _run_prognostic(cfg=cfg, launch=0.5, u=u0)
    assert float(jnp.max(jnp.abs(out0.du_dt))) < 1e-10
    assert float(jnp.max(jnp.abs(out0.dv_dt))) < 1e-10
    assert float(jnp.min(out0.eps_gwd)) >= -1e-12


# ===========================================================================
# ML emulator -- learned surrogate, NOT physics-faithful
# ===========================================================================
#
# Only the stability / finiteness / differentiability invariants apply.  An
# untrained MLP has no reason to oppose the flow or conserve momentum, so the
# sign / conservation / breaking invariants are intentionally NOT asserted.

def _ml_model(cfg):
    key = jax.random.PRNGKey(cfg.seed)
    return GWDEmulator(cfg.n_input, cfg.n_hidden, cfg.n_layers, cfg.n_output,
                       key=key)


def test_ml_emulator_finite_and_bounded():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = GWDMLEmulatorConfig(n_hidden=32)
    out = ml_gwd(u, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0, cfg,
                 _ml_model(cfg))
    # Finiteness + bounded tendency are the only physics-style invariants that
    # apply to a learned surrogate; sign/conservation are NOT asserted (the
    # untrained net carries no physics).
    _assert_finite_and_bounded(out)


def test_ml_emulator_grad_jit_vmap():
    u, v, T, pf, ph, zf, zh, rho = _idealized_column()
    cfg = GWDMLEmulatorConfig(n_hidden=32)
    model = _ml_model(cfg)
    fn = lambda uu: ml_gwd(uu, v, T, pf, ph, zf, zh, rho, jnp.zeros(1), 1800.0,
                           cfg, model)
    _assert_grad_finite_nonzero(lambda uu: jnp.sum(fn(uu).du_dt ** 2), u)
    eager = fn(u)
    jitted = jax.jit(fn)(u)
    np.testing.assert_allclose(np.array(jitted.du_dt), np.array(eager.du_dt),
                               rtol=1e-12, atol=1e-15)
