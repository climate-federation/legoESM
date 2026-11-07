"""Scheme-design tests for ``lindzen_gwd`` (E3SM v3.0.1 orographic-only port).

The term-by-term oracle match (E3SM Fortran, max rel err <= 1e-10 at the
hard-limit smoothing) is ``test_lindzen_e3sm_oracle.py``.  This file pins the
invariants that must hold for any input: the vector momentum sink, the
discrete kinetic-energy heating closure, zero tendency without orography,
the fcrit2 response, gradient safety at the small-N / U -> 0 traps and the
dispatch hardening.  (The former helper-form pins of the surrogate launch and
saturation formulas were removed with the surrogate; the oracle test replaces
them.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
    LindzenConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import get_gwd_fn
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import (
    __physics_contract__,
    lindzen_gwd,
)

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# Limiters effectively OFF: isolates the Lindzen saturation-divergence / launch forms from the
# post-flux tndmax/umcfac magnitude caps (which break the exact stress-divergence balance).
def _no_limiter_config(**kw):
    base = dict(tndmax_per_day=1.0e12, umcfac=1.0e9, fcrit2=1.0, Fr_sharpness=200.0)
    base.update(kw)
    return LindzenConfig(**base)


def _columns(ncol=3, nlev=16, u_sfc=25.0, u_top=25.0, v0=0.0, T_sfc=290.0, T_top=220.0,
             neutral=False):
    """Deterministic hydrostatic column; u varies linearly top->surface, v uniform.

    u_top can be < u_sfc (saturation shear) or negative (critical-level reversal). v0=0 keeps the
    source direction along +x so U_proj == u. ``neutral=True`` builds an ADIABATIC profile
    (T ∝ (p/p_sfc)^κ ⇒ θ ≈ const ⇒ N → ~0) to exercise the safe_divide(1/N) trap.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    if neutral:
        T = T_sfc * (p_full / p_full[:, -1:]) ** constants.kappa      # adiabatic ⇒ N → ~0
    else:
        T = jnp.broadcast_to(jnp.linspace(T_top, T_sfc, nlev)[None, :], (ncol, nlev))
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    z_half_cumsum = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    # index 0 = top, -1 = surface
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.full((ncol, nlev), v0)
    lat = jnp.full((ncol,), 0.5)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


def _run(cfg, dt=300.0, h_topo_col=None, **kw):
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(**kw)
    out = lindzen_gwd(
        u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, cfg, h_topo_col=h_topo_col
    )
    return out, (u, v, T, p_full, p_half, z_full, z_half, rho, lat)


def test_lindzen_config_defaults():
    """Canary: the Lindzen closure constants (fcrit2=1 ⇒ cap at exact tau_sat)."""
    c = LindzenConfig()
    assert c.fcrit2 == 1.0
    assert c.h_topo == 500.0
    assert c.k_wave == 2.0 * np.pi / 100e3
    assert c.Fr_sharpness == 20.0
    assert c.crit_level_sharpness == 10.0
    assert c.crit_level_floor == 0.5
    assert c.tndmax_per_day == 500.0
    assert c.umcfac == 0.5


# ===========================================================================
# DESIGN: critical-level treatment (c = 0) + sign
# ===========================================================================
def test_lindzen_critical_level_absorption():
    """Critical level (c = 0): where the projected wind reverses, E3SM zeroes the
    saturation stress and deposits the residual in that layer; nothing propagates
    above it.  u goes +20 (surface) -> -8 (top).
    """
    out, inp = _run(LindzenConfig(), h_topo_col=jnp.full((3,), 800.0), u_sfc=20.0, u_top=-8.0)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    reversed_layer = u < 0.0
    assert reversed_layer.any()                                        # non-vacuity: crit level
    # No drag well above the critical level (the wave was absorbed below).
    k_crit = np.argmax(u[0] > 0.0)          # top-first: first positive level from the top
    assert np.allclose(du[:, : k_crit - 1], 0.0, atol=1e-14)
    assert np.any(np.abs(du[u > 0.0]) > 1e-10)
    # The applied drag is a vector sink (u*du + v*dv <= 0) everywhere.
    assert np.all(u * du + v * dv <= 1e-14)


def test_lindzen_sign_and_sink():
    """DESIGN: the VECTOR sink u·du_dt + v·dv_dt ≤ 0 at every level and eps_gwd ≥ 0.

    Uses an OBLIQUE wind (v0≠0) so the test genuinely covers the vector invariant — only the
    projection along the source-wind direction is guaranteed ≤ 0, NOT componentwise du_dt·u.
    """
    out, inp = _run(
        LindzenConfig(), h_topo_col=jnp.full((3,), 800.0), u_sfc=25.0, u_top=6.0, v0=9.0
    )
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    proj = u * du + v * dv                                             # vector projection
    assert np.all(proj <= 1e-14)
    assert np.any(proj < -1e-12)                                       # non-vacuity
    assert np.all(np.asarray(out.eps_gwd) >= 0.0)
    assert np.any(np.asarray(out.eps_gwd) > 1e-12)


# ===========================================================================
# KE->heat closure = the pointwise energy conservation (c=0 wave) + rest state
# ===========================================================================
def test_lindzen_energy_closure_ke_to_heat():
    """Discrete closure (E3SM gw_drag.F90:908-913): at every level
    c_pd*dT_dt == -[KE(u + dt*du) - KE(u)]/dt, so column enthalpy gain equals the
    resolved KE lost over the step; and it is non-negative (a sink).
    """
    dt = 300.0
    out, inp = _run(LindzenConfig(), dt=dt, h_topo_col=jnp.full((3,), 800.0),
                    u_sfc=25.0, u_top=6.0, v0=3.0)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    ke_loss = -(0.5 * ((u + dt * du) ** 2 + (v + dt * dv) ** 2) - 0.5 * (u ** 2 + v ** 2)) / dt
    heat = constants.c_pd * np.asarray(out.dT_dt)
    assert np.any(heat > 1e-9)
    np.testing.assert_allclose(heat, ke_loss, rtol=1e-9, atol=1e-15)
    assert np.all(heat >= -1e-15)


def test_lindzen_conserves_energy():
    """``conserves == ["energy"]``: a stationary (c = 0) wave carries no wave-energy
    flux and the discrete closure returns the resolved KE lost as local heat (see the
    closure test above).  Momentum is not conserved (sink to the mountain, leak
    through the model top where tau(top) != 0, as in E3SM).
    """
    assert __physics_contract__["conserves"] == ["energy"]


def test_lindzen_rest_state_zero_orography():
    """DESIGN idealized: zero subgrid orography (h=0) ⇒ zero launch ⇒ zero tendency."""
    out, _ = _run(LindzenConfig(), h_topo_col=jnp.zeros((3,)), u_sfc=25.0, u_top=6.0)
    assert np.allclose(np.asarray(out.du_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.dv_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.dT_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.eps_gwd), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "neutral,label",
    [(False, "stable + zero-top-wind (U->0 trap)"),
     (True, "near-neutral (N->0 trap: safe_divide(1,N) exercised)")],
)
def test_lindzen_ad_safe_at_traps(neutral, label):
    """NUMERICS: grad of column drag wrt a wind-scale is finite at the small-N and U→0 traps.

    The adiabatic column drives N down to the E3SM gw_prof floor (sqrt(n2min) = 1e-4), a
    large-but-finite 1/N cotangent; the stable column with u_top = 0 exercises the c = 0
    critical-level trap.
    """
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(
        u_sfc=20.0, u_top=0.0, neutral=neutral
    )
    N_full = np.asarray(brunt_vaisala_n_full(T, p_full, z_full))
    if neutral:
        assert np.max(N_full) < 5e-3                     # near-neutral: small-N (1/N) trap stressed
    cfg = LindzenConfig()
    h_col = jnp.full((3,), 600.0)

    def col_drag(scale):
        out = lindzen_gwd(
            u * scale, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg,
            h_topo_col=h_col,
        )
        return jnp.sum(out.du_dt ** 2)

    g = jax.grad(col_drag)(1.0)
    assert np.isfinite(float(g))


def test_dispatch_unknown_gwd_raises():
    """Dispatch hardening: an unknown GWD scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown GWD scheme"):
        get_gwd_fn(GravityWaveDragConfig(scheme="not_a_scheme"))


def test_fcrit2_scales_the_cap_not_the_activation():
    """Oracle-semantics regression (GWD recon lindzen #1): fcrit2 must scale
    the saturation CAP VALUE (effkwv semantics) — under the OLD activation
    wiring fcrit2=0.5 was provably inert wherever tau_carry <= tau_sat (the
    min-guard proof) and changed max accel by only ~1.4% on the recon's
    discriminating column vs 50% for the true cap scaling.  Pin: on a
    supersaturated launch column, halving fcrit2 must halve the transmitted
    (post-breaking) stress scale => substantially increase the deposited
    drag, far beyond the old wiring's percent-level response."""
    col = dict(ncol=1, u_sfc=20.0, u_top=20.0)
    h_topo = jnp.full((1,), 900.0)
    out_1, _ = _run(_no_limiter_config(fcrit2=1.0), h_topo_col=h_topo, **col)
    out_h, _ = _run(_no_limiter_config(fcrit2=0.5), h_topo_col=h_topo, **col)
    a1 = float(jnp.max(jnp.abs(out_1.du_dt)))
    ah = float(jnp.max(jnp.abs(out_h.du_dt)))
    assert ah != a1
    rel = abs(ah - a1) / a1
    assert rel > 0.10, (
        f"fcrit2 response {rel:.1%} — cap-scaling gives O(50%), the old "
        "activation wiring gave ~1.4% (dead-knob regression)")
