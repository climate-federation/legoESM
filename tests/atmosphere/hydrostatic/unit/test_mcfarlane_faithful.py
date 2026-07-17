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
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import (
    __physics_contract__,
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
    """KE→heat tie-back = the pointwise energy closure (``conserves=["energy"]``).

    ``dT_dt`` is computed as ``-(u·du+v·dv)/c_pd`` from the FINAL applied tendency
    and ``eps_gwd`` as the same column KE loss with the same ``ρ·dz``, so
    ``c_pd·∫ρ (dT/dt) dz == eps_gwd`` holds BY CONSTRUCTION.  For a c=0 wave (zero
    wave-energy flux) that by-construction identity IS the exact resolved-energy
    conservation the contract declares — this test locks it against accidental
    desync (a future edit changing one term but not the other).  It is a
    consistency/regression pin of the conservation closure, not an independent
    re-derivation of it (and NOT an E3SM-faithfulness claim).
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


# ---------------------------------------------------------------------------
# Conservation disclosure (matches the Hines/Lindzen launched-wave precedent)
# ---------------------------------------------------------------------------

def test_mcfarlane_conserves_energy():
    """``conserves == ["energy"]`` — a STATIONARY orographic wave (c=0) carries
    ZERO vertical wave-energy flux (F_E = c·F_momentum = 0, Eliassen–Palm), so
    all mean-flow KE removed is returned LOCALLY as heat and resolved KE +
    internal energy is conserved pointwise (``dT_dt`` from the FINAL tendency ⇒
    ``c_pd*sum(rho*dT*dz)==eps_gwd`` exactly; see the tie-back test below).

    The discriminator is "does the launched wave carry vertical energy flux?":
    c=0 orographic (this scheme, ``rayleigh`` direct-drag) ⇒ ``["energy"]``;
    c≠0 launched spectra (``hines``, ``prognostic_spectral``) whose waves carry
    energy the limiter discards ⇒ ``["none"]``. MOMENTUM is not conserved (sink
    to the subgrid mountain / critical level) — the critical-level radiation and
    the tendency limiter break MOMENTUM, not energy, closure.
    """
    assert __physics_contract__["conserves"] == ["energy"]


def test_mcfarlane_vector_ke_sink_not_componentwise_for_veering_wind():
    """The guaranteed KE-sink invariant is the VECTOR ``u·du+v·dv ≤ 0``, NOT the
    componentwise ``du_dt·u ≤ 0`` (which the pre-audit sign_convention claimed).

    The drag is directed along the FIXED surface-wind direction
    ``(cos_a, sin_a)``; for a wind that VEERS with height a deposition level can
    have its local ``u`` anti-parallel to that fixed direction, so
    ``du_dt·u = accel·cos_a·u`` flips POSITIVE there while the vector projection
    ``u·du_dt + v·dv_dt = accel·U_proj`` stays ≤ 0 (``U_proj > 0`` at any
    deposition level by the hard mask, ``accel ≤ 0``). This pins that the vector
    form is the correct invariant and the componentwise form is NOT.
    """
    ncol, nlev = 1, 24
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
    # Surface wind (15, 15) m/s (source dir = 45 deg); VEER so u flips negative
    # aloft while v grows -> U_proj stays > 0 (wave still propagates & deposits).
    u = jnp.broadcast_to(jnp.linspace(-10.0, 15.0, nlev)[None, :], (ncol, nlev))
    v = jnp.broadcast_to(jnp.linspace(25.0, 15.0, nlev)[None, :], (ncol, nlev))
    lat = jnp.full((ncol,), 0.5)
    cfg = McFarlaneConfig(h_topo=1500.0)
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg)

    # (a) VECTOR invariant holds at every level.
    ke_rate = u * out.du_dt + v * out.dv_dt
    assert jnp.all(ke_rate <= 1e-12), (
        f"vector KE sink violated: max(u*du+v*dv)={float(jnp.max(ke_rate)):.3e}"
    )
    # (b) NON-VACUOUS: at some deposition level the COMPONENTWISE du_dt*u is > 0,
    # so the componentwise claim is genuinely false for this veering column.
    comp = out.du_dt[0] * u[0]
    active = jnp.abs(out.du_dt[0]) > 1e-12
    assert jnp.any(active & (comp > 1e-10)), (
        "veering column did not produce a componentwise du_dt*u > 0 -> test "
        "vacuous; strengthen the veer so a deposition level has u anti-parallel "
        "to the surface-wind direction"
    )


# ---------------------------------------------------------------------------
# E3SM hdsp = 2*sgh displacement (config.use_e3sm_hdsp; gw_oro.F90:117,166-168)
# ---------------------------------------------------------------------------

def test_e3sm_hdsp_three_regime_partition_on_launch_stress():
    """Oracle displacement (hdsp = 2*sgh) vs legacy, on the launch helper:

    * below BOTH caps (4h² < fcrit2(U/N)²): exactly 4x the legacy stress;
    * above OUR cap (fcrit2(U/N)² <= h²): both Froude-limited — EQUAL;
    * in the band h² < fcrit2(U/N)² < 4h²: strictly between 1x and 4x
      (the flag hits the cap, legacy does not).
    (Skeptic-corrected three-regime partition of the GWD recon.)"""
    cfg_off = McFarlaneConfig()
    cfg_on = McFarlaneConfig(use_e3sm_hdsp=True)
    rho, N, U = jnp.array([1.2]), jnp.array([1.0e-2]), jnp.array([15.0])
    cap = float(cfg_off.fcrit2 * (U[0] / N[0]) ** 2)  # 2.25e6 m^2 (h=1500)

    def tau(cfg, h):
        return float(_mcfarlane_launch_stress(rho, N, U, jnp.array([h * h]), cfg)[0])

    h_small = 300.0                       # 4h^2 = 3.6e5 < cap
    assert abs(tau(cfg_on, h_small) / tau(cfg_off, h_small) - 4.0) < 1e-9

    h_huge = 5000.0                       # h^2 = 2.5e7 > cap
    t_on, t_off = tau(cfg_on, h_huge), tau(cfg_off, h_huge)
    assert abs(t_on - t_off) <= 1e-12 * t_off, "above the cap both must be Froude-limited"

    h_band = 1000.0                       # h^2 = 1e6 < cap = 2.25e6 < 4h^2 = 4e6
    r = tau(cfg_on, h_band) / tau(cfg_off, h_band)
    assert 1.0 < r < 4.0, f"band ratio {r} not in (1, 4)"
    # In the band the flagged stress IS the cap value.
    expected_cap = float(cfg_on.G_0 * rho[0] * N[0] * cfg_on.k_wave * cap * U[0])
    assert abs(tau(cfg_on, h_band) - expected_cap) <= 1e-9 * expected_cap


def test_e3sm_hdsp_matches_gw_oro_src_formula():
    """Flag ON, below the cap: tau_0 == oroko2*min(hdsp^2, sghmax)*rho*N*U
    with hdsp = 2*sgh, oroko2 = 0.5*k (gw_oro.F90:33,117,166-168) — the E3SM
    gw_oro_src launch, hand-evaluated."""
    cfg = McFarlaneConfig(use_e3sm_hdsp=True)
    assert cfg.G_0 == 0.5  # oroko2 = 0.5*kwv folds into G_0
    rho, N, U = jnp.array([1.1]), jnp.array([1.2e-2]), jnp.array([11.0])
    sgh = 250.0
    hdsp_sq = (2.0 * sgh) ** 2
    sghmax = float(cfg.fcrit2 * (U[0] / N[0]) ** 2)
    expected = float(
        cfg.G_0 * cfg.k_wave * min(hdsp_sq, sghmax) * rho[0] * N[0] * U[0]
    )
    got = float(_mcfarlane_launch_stress(rho, N, U, jnp.array([sgh * sgh]), cfg)[0])
    assert abs(got - expected) <= 1e-12 * expected


def test_e3sm_hdsp_default_off_bit_identical():
    """Canary: the flag defaults False and the helper is bit-identical to the
    legacy form there (regression pin for the default path)."""
    assert McFarlaneConfig().use_e3sm_hdsp is False
    cfg = McFarlaneConfig()
    rho, N, U = jnp.array([1.2]), jnp.array([1.0e-2]), jnp.array([12.0])
    h_sq = jnp.array([90000.0])
    got = _mcfarlane_launch_stress(rho, N, U, h_sq, cfg)
    expected = cfg.G_0 * rho * N * cfg.k_wave * jnp.minimum(
        h_sq, cfg.fcrit2 * (U / N) ** 2
    ) * U
    assert jnp.array_equal(got, expected)


def test_e3sm_hdsp_requires_per_column_topo():
    """G6 guard: use_e3sm_hdsp on the scalar config.h_topo fallback raises
    (a quadrupled uniform pseudo-mountain would drag over ocean planet-wide;
    this scheme has no landfrac factor)."""
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _column()
    cfg = McFarlaneConfig(use_e3sm_hdsp=True)
    with pytest.raises(ValueError, match="use_e3sm_hdsp.*h_topo_col"):
        mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat,
                      1800.0, cfg, h_topo_col=None)
    # And with the per-column field it runs.
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat,
                        1800.0, cfg, h_topo_col=jnp.full((u.shape[0],), 300.0))
    assert bool(jnp.all(jnp.isfinite(out.du_dt)))
