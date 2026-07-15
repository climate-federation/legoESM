"""Faithfulness / invariant tests for the McFarlane (1987)-INSPIRED orographic GWD.

``mcfarlane.py`` is McFarlane-INSPIRED, NOT a faithful E3SM ``gw_oro`` port (use
``e3sm_cam`` for that — see the scheme docstring's "Faithfulness" section).  The
one piece genuinely shared with the oracle is the **Froude-capped launch-stress
FORM** ``tau_0 = G0·ρ·N·k·min(h², fcrit2·(U/N)²)·U`` (E3SM ``gw_oro_src``); these
tests lock that FORM directly on the exposed ``_mcfarlane_launch_stress`` helper
(the total deposited drag is downstream of the Lindzen ``tau_sat`` saturation and
cannot isolate the cap).  The remaining tests pin scheme INVARIANTS (strict
momentum sink, critical-level absorption, heating/KE diagnostic tie-back) — not
E3SM-faithfulness — and the documented 4× ``hdsp`` source-amplitude departure.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import (
    mcfarlane_gwd,
    _mcfarlane_launch_stress,
)


def _column(ncol=1, nlev=24, u_sfc=15.0, u_top=15.0):
    """Deterministic column (surface at index -1)."""
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = jnp.broadcast_to(jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev))
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0, None)))
    z_half = jnp.concatenate(
        [jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((ncol, 1))], axis=1,
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros_like(u)
    lat = jnp.full((ncol,), 0.5)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


# ---------------------------------------------------------------------------
# Froude-capped launch stress — the one E3SM gw_oro FORM (tested on the helper)
# ---------------------------------------------------------------------------

def test_mcfarlane_faithful_froude_cap_form_on_launch_stress():
    """``tau_0 = G0·ρ·N·k·min(h², fcrit2·(U/N)²)·U`` (E3SM ``gw_oro_src``).

    Tested DIRECTLY on ``_mcfarlane_launch_stress`` (not via deposited drag,
    which the Lindzen ``tau_sat`` saturation confounds): quadratic in ``h`` below
    the Froude cap, EXACTLY ``h``-independent above it, and equal to the analytic
    cap value ``G0·ρ·N·k·fcrit2·(U/N)²·U`` there.
    """
    cfg = McFarlaneConfig()
    rho, N, U = jnp.array([1.2]), jnp.array([1.0e-2]), jnp.array([15.0])
    froude_cap = float(cfg.fcrit2 * (U[0] / N[0]) ** 2)  # (1500)^2 = 2.25e6 m^2

    def tau0(h):
        return float(_mcfarlane_launch_stress(rho, N, U, jnp.array([h * h]), cfg)[0])

    # Below the cap (h << 1500 m): tau_0 is exactly quadratic in h.
    assert abs(tau0(200.0) / tau0(100.0) - 4.0) < 1e-9, "not quadratic in h below cap"
    # Above the cap (h >> 1500 m): tau_0 is exactly h-independent.
    t_a, t_b = tau0(5000.0), tau0(10000.0)
    assert abs(t_b - t_a) <= 1e-12 * t_a, "launch stress not Froude-capped above cap"
    # And equals the analytic Froude-limited value.
    expected = float(cfg.G_0 * rho[0] * N[0] * cfg.k_wave * froude_cap * U[0])
    assert abs(t_a - expected) <= 1e-9 * expected, "Froude-cap value wrong"


def test_mcfarlane_faithful_froude_cap_scales_u3_over_n():
    """Above the cap the Froude-limited stress scales ``tau_0 ∝ U³ / N``.

    ``h_eff² = fcrit2·(U/N)²`` ⇒ ``tau_0 = G0·ρ·k·fcrit2·U³/N`` — the E3SM /
    McFarlane Froude signature.  Verified exactly on the launch-stress helper
    with a huge mountain (cap binds).
    """
    cfg = McFarlaneConfig()
    rho, h_huge_sq = jnp.array([1.2]), jnp.array([1.0e12])

    def tau0(U, N):
        return float(
            _mcfarlane_launch_stress(rho, jnp.array([N]), jnp.array([U]), h_huge_sq, cfg)[0]
        )

    # U^3: doubling the source wind octuples the stress.
    assert abs(tau0(20.0, 1e-2) / tau0(10.0, 1e-2) - 8.0) < 1e-9
    # 1/N: doubling N halves the stress.
    assert abs(tau0(15.0, 2e-2) / tau0(15.0, 1e-2) - 0.5) < 1e-9


def test_mcfarlane_launch_prefactor_and_hdsp_departure():
    """``G_0 = 0.5`` folds ``oroko2 = 0.5·k``; DOCUMENTED 4× ``hdsp`` departure.

    Below the cap ``tau_0 = G0·ρ·N·k·h²·U`` exactly.  E3SM forms the displacement
    ``hdsp = 2·sgh`` and launches ``0.5·k·hdsp²``, so at equal ``sgh`` (and below
    the Froude cap, where ``h²`` enters) E3SM's stress is 4× this scheme's —
    pinned here so the departure is explicit, not a silent bug (fixing it is
    behavioral → RCE-gated; see the module docstring).
    """
    cfg = McFarlaneConfig()
    assert cfg.G_0 == 0.5
    rho, N, U = jnp.array([1.2]), jnp.array([1.0e-2]), jnp.array([12.0])
    h = 100.0  # below the Froude cap (~1.8 km here)
    tau0 = float(_mcfarlane_launch_stress(rho, N, U, jnp.array([h * h]), cfg)[0])
    expected = float(cfg.G_0 * rho[0] * N[0] * cfg.k_wave * h * h * U[0])
    assert abs(tau0 - expected) <= 1e-9 * expected, "launch amplitude != G0*rho*N*k*h^2*U"
    # E3SM hdsp = 2*sgh would launch 4x more at the same sgh=h.
    tau0_hdsp = float(
        _mcfarlane_launch_stress(rho, N, U, jnp.array([(2.0 * h) ** 2]), cfg)[0]
    )
    assert abs(tau0_hdsp / tau0 - 4.0) < 1e-9, "hdsp=2*sgh departure factor is not 4"


# ---------------------------------------------------------------------------
# Scheme invariants (NOT E3SM-faithfulness)
# ---------------------------------------------------------------------------

def test_mcfarlane_strict_momentum_sink():
    """Invariant: the drag never adds KE (``u·du/dt + v·dv/dt ≤ 0`` everywhere).

    Deposited drag is a deceleration along the source wind — a genuine momentum
    sink at every level.  (An implementation invariant, not an oracle match.)
    """
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column(
        nlev=24, u_sfc=30.0, u_top=5.0,
    )
    cfg = McFarlaneConfig(h_topo=1000.0)
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg)
    ke_rate = u * out.du_dt + v * out.dv_dt
    assert jnp.all(ke_rate <= 1e-12), (
        f"drag added KE: max(u*du+v*dv)={float(jnp.max(ke_rate)):.3e}"
    )


def test_mcfarlane_critical_level_no_reversed_acceleration():
    """Invariant: critical-level absorption never accelerates the reversed flow.

    Across a source-projected-wind reversal the hard ``U_proj > 0`` mask makes
    the deposited drag EXACTLY zero where ``U_proj < 0``, so the single-wave drag
    never accelerates the flow it passed through (``u·du/dt + v·dv/dt ≤ 0``).
    NOTE: this is the scheme's RADIATING single-wave treatment, NOT E3SM's
    crossing-layer deposition — an invariant, not an oracle match.
    """
    # Surface wind +25 m/s (source), reversing to -15 m/s aloft; v=0 ⇒ U_proj=u.
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column(
        nlev=24, u_sfc=25.0, u_top=-15.0,
    )
    cfg = McFarlaneConfig(h_topo=1000.0)
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg)
    ke_rate = u * out.du_dt + v * out.dv_dt
    assert jnp.all(ke_rate <= 1e-12), (
        f"drag accelerated reversed flow: max(u*du+v*dv)={float(jnp.max(ke_rate)):.3e}"
    )
    # Where U_proj < 0 the hard mask zeroes BOTH tendency components exactly.
    u_proj = u[0]  # v = 0 in this column
    reversed_lyr = u_proj < 0.0
    assert jnp.all(jnp.abs(out.du_dt[0])[reversed_lyr] == 0.0)
    assert jnp.all(jnp.abs(out.dv_dt[0])[reversed_lyr] == 0.0)


def test_mcfarlane_ke_heat_diagnostic_consistency():
    """Diagnostic tie-back regression (NOT an independent energy proof).

    ``dT_dt`` is DEFINED as ``-(u·du+v·dv)/c_pd`` and ``eps_gwd`` as the same
    column KE loss with the same ``ρ·dz``, so ``c_pd·∫ρ (dT/dt) dz == eps_gwd``
    holds BY CONSTRUCTION.  This locks that imposed heating/KE tie-back against
    accidental desync (e.g. a future edit changing one but not the other); it is
    a consistency regression, not evidence of energy conservation or faithfulness.
    """
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column(
        nlev=24, u_sfc=25.0, u_top=8.0,
    )
    cfg = McFarlaneConfig(h_topo=800.0)
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg)
    dz = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    heat_col = constants.c_pd * jnp.sum(rho * out.dT_dt * dz, axis=1)
    assert jnp.allclose(heat_col, out.eps_gwd, rtol=1e-6, atol=1e-10)
    assert float(out.eps_gwd[0]) >= 0.0, "eps_gwd (column KE loss) must be >= 0"
