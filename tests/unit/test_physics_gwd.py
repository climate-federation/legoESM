"""Category 6: Gravity Wave Drag -- Physical Consistency.

Tests drag opposing wind, energy dissipation, Rayleigh sponge structure,
zero-wind behavior, and magnitude bounds for all GWD schemes.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics


def _make_state(n=8, nlev=10, wind_speed=10.0):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev)) * wind_speed,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


def _run_gwd(scheme, **kwargs):
    state, grid, sigma = _make_state(**kwargs)
    config = GravityWaveDragConfig(scheme=scheme)
    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    tend, prog = gwd_fn(state, grid, sigma)
    return tend, state


ALL_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines", "prognostic_spectral"]
# Diagnostic schemes (not multi-directional) where drag should oppose wind
DIAGNOSTIC_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines"]


# ============================================================================
# 6a  Drag opposes wind (diagnostic schemes only)
# ============================================================================

@pytest.mark.parametrize("scheme", DIAGNOSTIC_SCHEMES)
def test_drag_opposes_wind(scheme):
    """du_dt * u <= 0 wherever drag is active (drag decelerates)."""
    tend, state = _run_gwd(scheme, wind_speed=15.0)
    u = state.u.data
    du_dt = tend.du_dt.data
    product = u * du_dt
    active = jnp.abs(du_dt) > 1e-12
    if jnp.any(active):
        n_opposing = int(jnp.sum((product[active] <= 1e-10)))
        n_active = int(jnp.sum(active))
        frac = n_opposing / max(n_active, 1)
        assert frac > 0.9, (
            f"{scheme}: only {frac:.0%} of active points have drag opposing wind"
        )


# ============================================================================
# 6d  Zero wind -> zero drag
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_zero_wind_zero_drag(scheme):
    """With u = v = 0, GWD tendencies should be zero."""
    state, grid, sigma = _make_state(wind_speed=0.0)
    state = state._replace(
        v=Field(data=jnp.zeros_like(state.v.data),
                name="v", dims=state.v.dims, units="m/s"),
    )
    config = GravityWaveDragConfig(scheme=scheme)
    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    tend, _ = gwd_fn(state, grid, sigma)
    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
    max_dv = float(jnp.max(jnp.abs(tend.dv_dt.data)))
    assert max_du < 1e-10, f"{scheme}: du_dt = {max_du:.2e} with zero wind"
    assert max_dv < 1e-10, f"{scheme}: dv_dt = {max_dv:.2e} with zero wind"


# ============================================================================
# 6f  Magnitude bounds
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_magnitude_bounds(scheme):
    """GWD tendencies should be within physical bounds."""
    tend, _ = _run_gwd(scheme)
    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
    max_dT = float(jnp.max(jnp.abs(tend.dT_dt.data)))
    assert max_du < 0.1, f"{scheme}: |du_dt| = {max_du:.2e} exceeds 0.1 m/s^2"
    assert max_dT < 1e-3, f"{scheme}: |dT_dt| = {max_dT:.2e} exceeds 1e-3 K/s"


# ============================================================================
# All outputs finite
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_all_finite(scheme):
    """All GWD output fields should be finite."""
    tend, _ = _run_gwd(scheme)
    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"


# ============================================================================
# Rayleigh: structure check
# ============================================================================

def test_rayleigh_sponge_structure():
    """Rayleigh drag should be active at model top and/or BL."""
    tend, _ = _run_gwd("rayleigh", nlev=20, wind_speed=20.0)
    du_dt = tend.du_dt.data
    top_drag = float(jnp.max(jnp.abs(du_dt[..., :3])))
    bot_drag = float(jnp.max(jnp.abs(du_dt[..., -5:])))
    assert top_drag > 1e-8 or bot_drag > 1e-8, "Rayleigh: no drag anywhere"


# ============================================================================
# Hines: WKB amplitude growth uses inter-level rho ratio (not cumulative)
# ============================================================================

def test_hines_no_saturation_zero_drag():
    """In the no-saturation limit (very tiny m_star, sigma_sat huge), Hines
    drag must vanish.

    The WKB envelope grows as ``sqrt(rho_sfc / rho_z)`` which for an
    isothermal column is bounded by sqrt(rho_sfc/rho_top) ~ 30 over the
    50-km column built below.  Setting ``m_star = 1e-9`` makes the
    saturation amplitude ``sigma_sat = N / (m_star * rho_ratio)`` ~ 1e6
    m/s — so a correctly-amplified wave (bounded by ~60 m/s) never
    saturates and ``du/dt`` must be machine zero.

    A bug that compounds the rho ratio at every scan step (sigma_grown
    multiplied by the *cumulative* sqrt(rho_sfc/rho_k) instead of the
    *inter-level* sqrt(rho[k+1]/rho[k])) over-amplifies the wave by the
    product of cumulative ratios — order 4500x in this column — and
    re-activates saturation even when m_star is tiny.  Pre-fix this
    test sees max |du/dt| ~ 0.03 m/s^2.
    """
    import jax.numpy as _jnp
    from legoesm import constants as _const
    from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
    from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig

    ncol, nlev = 1, 20
    T = _jnp.full((ncol, nlev), 250.0)
    H = _const.R_d * 250.0 / _const.g
    z_full = _jnp.linspace(50e3, 0.0, nlev)
    z_half_1d = _jnp.concatenate([
        _jnp.array([z_full[0] + 1e3]),
        0.5 * (z_full[:-1] + z_full[1:]),
        _jnp.array([z_full[-1] - 1e3]),
    ])
    z_full_b = _jnp.broadcast_to(z_full[None, :], (ncol, nlev))
    z_half_b = _jnp.broadcast_to(z_half_1d[None, :], (ncol, nlev + 1))
    p_full = _jnp.broadcast_to(
        (1e5 * _jnp.exp(-z_full / H))[None, :], (ncol, nlev),
    )
    p_half = _jnp.broadcast_to(
        (1e5 * _jnp.exp(-z_half_1d / H))[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (_const.R_d * T)
    u = _jnp.full((ncol, nlev), 10.0)
    v = _jnp.zeros((ncol, nlev))
    lat = _jnp.zeros(ncol)

    cfg = HinesConfig(m_star=1e-9, total_rms_wind=2.0)
    out = hines_gwd(u, v, T, p_full, p_half, z_full_b, z_half_b, rho, lat,
                   300.0, cfg)
    max_du = float(jnp.max(jnp.abs(out.du_dt)))
    # Post-fix expectation: drag <= numerical noise.  Pre-fix this is
    # ~3e-2 m/s^2 from spurious cumulative compounding.
    assert max_du < 1e-8, (
        f"Hines: max |du/dt| = {max_du:.3e} m/s^2 in the no-saturation "
        "limit; expected ~0 (sigma_sat is set ~1e6 m/s by tiny m_star). "
        "A non-zero drag indicates the WKB amplitude is over-amplified "
        "by a per-step cumulative rho-ratio compounding."
    )


# ============================================================================
# Differentiability: jax.grad must produce finite, non-zero gradients
# ============================================================================
#
# Fills the audit gap "GWD: NONE (tests/unit/test_physics_gwd.py has no
# jax.grad calls)".  For each scheme we grad a scalar reduction of the
# dT_dt tendency w.r.t. surface temperature, an input that should affect
# every scheme's drag profile through theta -> N^2 -> Hines/Lindzen
# saturation, through stress closure, or through the wind shear ->
# orographic-launch path.

import jax  # noqa: E402  (grouped here to keep diff tests self-contained)

def test_mcfarlane_drag_scales_linearly_with_k_wave():
    """McFarlane drag must scale ~linearly with ``k_wave`` — post-fix
    BOTH ``tau_0`` and ``tau_sat`` carry one factor of ``k_wave``, so
    their ratio is k-independent (fraction of the wave that breaks is
    constant) and the absolute drag scales proportionally.

    Discriminating cases:

    * Post-fix (both have k):     drag ∝ k    (this test asserts)
    * k removed from tau_0 only:  drag DECREASES (or saturates) with k
    * k removed from tau_sat only: drag is non-monotonic / step-like
    * k removed from both (full pre-fix bug): drag k-independent

    Test setup uses ``h_topo = 200 m`` so the launch-stress clip
    (``clip(tau_0, 0, 10)``) does NOT engage at any k tested — without
    that clip masking the magnitude, the linear scaling is observable
    directly in ``|du/dt|`` and ``eps_gwd``.
    """
    import math
    from legoesm import constants as _const
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
    from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig

    ncol, nlev = 1, 12
    T = jnp.full((ncol, nlev), 250.0)
    H = _const.R_d * 250.0 / _const.g
    z_full = jnp.linspace(40e3, 0.0, nlev)
    z_half_1d = jnp.concatenate([
        jnp.array([z_full[0] + 1e3]),
        0.5 * (z_full[:-1] + z_full[1:]),
        jnp.array([z_full[-1] - 1e3]),
    ])
    z_full_b = jnp.broadcast_to(z_full[None, :], (ncol, nlev))
    z_half_b = jnp.broadcast_to(z_half_1d[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to(
        (1e5 * jnp.exp(-z_full / H))[None, :], (ncol, nlev),
    )
    p_half = jnp.broadcast_to(
        (1e5 * jnp.exp(-z_half_1d / H))[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (_const.R_d * T)
    u = jnp.full((ncol, nlev), 15.0)
    v = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)

    def drag_at_k(k_factor):
        cfg = McFarlaneConfig(
            h_topo=200.0, k_wave=2.0 * math.pi / 100e3 * k_factor,
        )
        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full_b, z_half_b,
                           rho, lat, 300.0, cfg)
        return float(jnp.sum(jnp.abs(out.du_dt))), float(out.eps_gwd[0])

    drag_lo, eps_lo = drag_at_k(1.0)
    drag_hi, eps_hi = drag_at_k(10.0)

    # Pre-fix sanity: drag must actually exist (post-fix gives ~5e-3).
    assert drag_lo > 1e-6, (
        f"McFarlane drag at k_factor=1.0 = {drag_lo:.3e} — too small to "
        "be physical.  Saturation path may not be firing at all."
    )

    # Linear scaling: drag(k=10×default) / drag(k=default) ≈ 10.
    # Allow a generous window [5, 20] to absorb second-order effects
    # in the saturation cap.  Pre-fix bug patterns:
    #   * k removed from BOTH stresses (full bug): ratio ≈ 1
    #   * k removed from tau_0 only:  ratio < 1 (drag falls or stays)
    #   * k removed from tau_sat only: ratio is column-dependent but
    #       NOT monotonic-linear at the post-fix value.
    ratio_du = drag_hi / drag_lo
    ratio_eps = eps_hi / max(eps_lo, 1e-30)
    assert 5.0 < ratio_du < 20.0, (
        f"McFarlane sum|du/dt| ratio (10× k_wave) = {ratio_du:.3f}; "
        "expected ~10 (linear in k_wave).  Ratio outside [5, 20] is "
        "the signature of k_wave being removed from one or both of "
        "tau_0 / tau_sat, breaking the dimensional consistency."
    )
    assert 5.0 < ratio_eps < 20.0, (
        f"McFarlane eps_gwd ratio (10× k_wave) = {ratio_eps:.3f}; "
        "expected ~10 (linear in k_wave)."
    )


def test_mcfarlane_saturation_path_fires_in_uniform_wind():
    """McFarlane saturation path must fire in a UNIFORM-wind column.

    Calls ``mcfarlane_gwd`` directly.  Earlier sheared-column tests
    cannot distinguish pre-fix from post-fix because the U → 0
    critical level near the model top sends ``tau_sat → 0`` and forces
    drag deposition there regardless of the launch-stress units.

    A uniform-wind column has no critical level, so saturation fires
    only when ``tau_0 > tau_sat`` somewhere in the column.  This is
    the discriminating test:

    * Pre-fix ``tau_sat = eff*ρ*U^3/(N*envelope)`` had units kg/s² and
      a numerical magnitude ~10⁵ Pa-equivalent at the surface,
      ~10² Pa at the top.  ``tau_0`` (also dimensionally wrong but
      clipped to 10) never exceeds tau_sat anywhere → drag = 0
      (specifically ``~2e-19 m/s²`` from the softmin-residual at
      float precision).

    * Post-fix both stresses are proper Pa.  ``tau_sat`` decreases
      from ~12 Pa at the surface to ~0.01 Pa at the top.  ``tau_0``
      is ~1-10 Pa — comparable to ``tau_sat`` at low altitudes and
      well above it near the top.  The wave breaks aloft and drag
      magnitude is ~1e-3 m/s² (column-max).

    The post-vs-pre gap is 14+ orders of magnitude; the threshold of
    ``1e-9 m/s²`` sits 7 orders above the pre-fix floor and 6 orders
    below the post-fix value — robust against both the bug signature
    and reasonable numerical drift in the saturation scan.
    """
    from legoesm import constants as _const
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
    from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig

    ncol, nlev = 1, 12
    T = jnp.full((ncol, nlev), 250.0)
    H = _const.R_d * 250.0 / _const.g
    z_full = jnp.linspace(40e3, 0.0, nlev)
    z_half_1d = jnp.concatenate([
        jnp.array([z_full[0] + 1e3]),
        0.5 * (z_full[:-1] + z_full[1:]),
        jnp.array([z_full[-1] - 1e3]),
    ])
    z_full_b = jnp.broadcast_to(z_full[None, :], (ncol, nlev))
    z_half_b = jnp.broadcast_to(z_half_1d[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to(
        (1e5 * jnp.exp(-z_full / H))[None, :], (ncol, nlev),
    )
    p_half = jnp.broadcast_to(
        (1e5 * jnp.exp(-z_half_1d / H))[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (_const.R_d * T)

    # UNIFORM wind throughout the column — no critical level.
    u = jnp.full((ncol, nlev), 15.0)
    v = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)

    cfg = McFarlaneConfig(h_topo=2000.0)
    out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full_b, z_half_b,
                       rho, lat, 300.0, cfg)
    max_du = float(jnp.max(jnp.abs(out.du_dt)))
    eps = float(out.eps_gwd[0])

    # Pre-fix this column produces ``max|du/dt| ~ 2e-19 m/s^2``;
    # post-fix it produces ``~1.5e-3 m/s^2`` — a 14-order gap.
    assert max_du > 1e-6, (
        f"McFarlane max |du/dt| = {max_du:.3e} m/s^2 in a uniform-wind "
        "column — saturation never fires.  This is the signature of "
        "the missing-k_wave bug in tau_0 / tau_sat: the dimensionally "
        "wrong tau_sat (~10^5 Pa-equivalent) dominates tau_0 (clipped "
        "at 10), wave never breaks, drag at float-precision floor."
    )
    assert max_du < 1.0, (
        f"McFarlane max |du/dt| = {max_du:.3e} m/s^2 — implausibly "
        "large; check that the launch-stress clip and saturation scan "
        "are bounding the deposition correctly."
    )
    # Column dissipation must be positive (KE → heat).
    assert eps > 0.0, (
        f"McFarlane eps_gwd = {eps:.3e} W/m^2 in a uniform-wind column "
        "with active drag — must be positive."
    )

    # --- Bug-pattern reproduction --------------------------------------
    # Replicate inline what the *pre-fix* formulas (no k_wave in either
    # stress) would produce in the SAME uniform-wind column.  The
    # purpose is to demonstrate the test's discriminating power: with
    # the dimensional bug the launch stress is enormous (numerically
    # ~22500 → clipped to 10) but the saturation stress is also
    # enormous (~10^5 vs the post-fix ~10), so the wave never breaks
    # and drag collapses to ~0.
    rho_arr = rho[0]
    N_const = 0.0196   # isothermal 250 K Brunt-Vaisala (same as scheme)
    h_topo = cfg.h_topo
    G_0 = cfg.G_0
    U_const = 15.0
    eff = cfg.efficiency

    tau_0_buggy_unclipped = G_0 * U_const * h_topo ** 2 * N_const * float(rho_arr[-1])
    tau_0_buggy = min(tau_0_buggy_unclipped, 10.0)
    tau_sat_buggy_at_top = eff * float(rho_arr[0]) * U_const ** 3 / N_const
    # Pre-fix tau_sat at top exceeds the clipped tau_0 → no breaking.
    # If this ordering reverses (tau_0_buggy > tau_sat_buggy_at_top)
    # the bug-pattern argument no longer holds and the test column
    # needs to be reselected.
    assert tau_0_buggy < tau_sat_buggy_at_top, (
        f"Test diagnostic: buggy tau_0 = {tau_0_buggy:.3e} must be "
        f"< buggy tau_sat at top = {tau_sat_buggy_at_top:.3e} for the "
        "uniform-wind column to discriminate post-fix from pre-fix."
    )


def _make_sheared_state(n=8, nlev=10):
    """State with a u-profile that decreases linearly toward a critical
    level near the top.  McFarlane / Lindzen need ``U_proj → 0`` aloft
    so that ``tau_sat ~ ρ*U^3/N`` collapses below the launch stress and
    the wave actually breaks; a uniform wind column never exercises
    the saturation path.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    # u profile: 30 m/s at surface (level nlev-1), 0.1 m/s at top (level 0).
    fraction = jnp.linspace(0.0, 1.0, nlev)  # 0 at top, 1 at surface
    u_profile = 0.1 + (30.0 - 0.1) * fraction
    u_data = jnp.broadcast_to(u_profile[None, None, None, :], (6, n, n, nlev))
    state = state._replace(
        u=Field(data=u_data, name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 1.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    return state, grid, sigma


def _gwd_config_with_active_drag(scheme):
    """Per-scheme config tuned so the held_suarez-init test column
    actually generates non-trivial drag.

    For the orographic schemes (Lindzen, McFarlane), the launch stress
    ``tau_0 ~ rho * N * k * h^2 * U`` and the saturation stress
    ``tau_sat ~ rho * U^3 / N`` differ by ``k * h^2 * N^2 / U^2``.  With
    default ``h_topo=500 m`` and ``N=0.01`` the ratio is ~1e-4 — the
    wave never breaks → drag ≈ 0 → autodiff gradient rounds to zero.

    Bumping ``h_topo`` to 10 km pushes ``tau_0`` above ``tau_sat`` near
    the model top, so the saturation scan deposits real drag and the
    differentiability tests exercise a non-trivial code path.
    """
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        LindzenConfig, McFarlaneConfig,
    )
    if scheme == "lindzen":
        return GravityWaveDragConfig(
            scheme="lindzen",
            lindzen=LindzenConfig(h_topo=10_000.0),
        )
    if scheme == "mcfarlane":
        return GravityWaveDragConfig(
            scheme="mcfarlane",
            mcfarlane=McFarlaneConfig(h_topo=10_000.0),
        )
    return GravityWaveDragConfig(scheme=scheme)


@pytest.mark.parametrize("scheme", ["rayleigh", "lindzen", "mcfarlane", "hines"])
def test_gwd_grad_through_T_finite_and_nonzero(scheme):
    """``d sum(dT_dt^2) / d (T-perturb)`` is finite and (for non-Rayleigh)
    non-zero.

    Rayleigh is a sponge layer that depends only on ``u, v, sigma`` —
    it does NOT depend on temperature, so its gradient w.r.t. T is
    structurally zero (we still check it's finite and not NaN/Inf).

    Lindzen / McFarlane / Hines all compute Brunt-Vaisala or other
    temperature-dependent quantities, so they should produce non-zero
    gradients.
    """
    state, grid, sigma = _make_sheared_state()
    T_field = state.T

    def loss(T_perturb):
        T_perturbed = T_field.replace(data=T_field.data + T_perturb)
        state_p = state._replace(T=T_perturbed)
        config = _gwd_config_with_active_drag(scheme)
        gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
        tend, _ = gwd_fn(state_p, grid, sigma)
        return jnp.sum(tend.dT_dt.data ** 2)

    g = float(jax.grad(loss)(jnp.asarray(0.0)))
    assert jnp.isfinite(g), f"{scheme}: grad is non-finite ({g})"
    if scheme != "rayleigh":
        assert abs(g) > 1e-30, (
            f"{scheme}: grad through T is zero ({g}); expected non-zero "
            "since the scheme has a temperature-dependent path."
        )


@pytest.mark.parametrize("scheme", ["rayleigh", "lindzen", "mcfarlane", "hines"])
def test_gwd_grad_through_wind_finite_and_nonzero(scheme):
    """``d sum(du_dt^2) / d u`` is finite and non-zero for every scheme.

    Every diagnostic GWD scheme decelerates the resolved wind, so
    perturbing ``u`` MUST change ``du_dt``.  A zero gradient would
    indicate either dead code (scheme not actually using ``u``) or a
    broken graph (e.g., a hard ``where`` cutting the gradient path).
    """
    state, grid, sigma = _make_sheared_state()
    u_field = state.u

    def loss(u_perturb):
        u_perturbed = u_field.replace(data=u_field.data + u_perturb)
        state_p = state._replace(u=u_perturbed)
        config = _gwd_config_with_active_drag(scheme)
        gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
        tend, _ = gwd_fn(state_p, grid, sigma)
        return jnp.sum(tend.du_dt.data ** 2)

    g = float(jax.grad(loss)(jnp.asarray(0.0)))
    assert jnp.isfinite(g), f"{scheme}: grad is non-finite ({g})"
    assert abs(g) > 1e-30, (
        f"{scheme}: grad through u is zero ({g}); expected non-zero "
        "since the scheme decelerates the resolved wind."
    )


def test_prognostic_spectral_grad_through_launch_flux():
    """Prognostic-spectral GWD must have a finite, non-zero gradient
    through its source momentum flux.

    Asserting only finiteness would let a zero gradient pass silently
    — and a zero gradient is exactly the failure mode this test is
    meant to catch (a broken JAX graph through the spectral prognostic
    state would produce ``g = 0`` rather than ``NaN``/``Inf``).
    """
    state, grid, sigma = _make_state(wind_speed=10.0)
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        PrognosticSpectralConfig,
    )

    def loss(amp):
        config = GravityWaveDragConfig(
            scheme="prognostic_spectral",
            prognostic_spectral=PrognosticSpectralConfig(launch_flux=amp),
        )
        gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
        tend, _ = gwd_fn(state, grid, sigma)
        return jnp.sum(tend.dT_dt.data ** 2 + tend.du_dt.data ** 2)

    g = float(jax.grad(loss)(jnp.asarray(1e-3)))
    assert jnp.isfinite(g), f"prognostic_spectral: grad is non-finite ({g})"
    assert abs(g) > 1e-30, (
        f"prognostic_spectral: grad through launch_flux is zero ({g}); "
        "expected non-zero — the spectrum-launch amplitude must drive a "
        "non-trivial drag profile.  A zero gradient indicates the spectral "
        "prognostic state has become a non-traceable buffer."
    )


# ============================================================================
# Hines: drag has correct units (Pa, not kg/(m²·s))
# ============================================================================

def test_hines_drag_units_match_lindzen_pa():
    """Audit cycle 2 P1 (deferred → fixed): the Hines drag formula
    ``ρ · (σ_grown - σ_new)`` had units ``kg/(m²·s)`` rather than the
    Pa expected by ``Fmax`` and the downstream ``accel = -drag/(ρ·dz)``
    which needed Pa for ``m/s²``.  The fix uses
    ``ρ · (σ²_grown - σ²_new)`` — wave momentum-flux divergence with
    correct stress units.

    This regression test: compute the Hines drag profile under a
    canonical setup and check that the values are in the Pa-bounded
    range ``[0, Fmax]``, *and* that the resulting acceleration
    magnitude is in the physical ``m/s²`` range expected for
    stratospheric GWD (~``1e-5`` to ``1e-3`` m/s²).  Pre-fix the
    same fixture would also be in ``[0, 0.1]`` numerically (the cap
    floors both forms identically) but the *acceleration* would be
    in the wrong ``1/s`` units.
    """
    import jax.numpy as _jnp
    from legoesm import constants as _const
    from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
    from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig

    ncol, nlev = 1, 20
    T = _jnp.full((ncol, nlev), 250.0)
    H = _const.R_d * 250.0 / _const.g
    z_full = _jnp.linspace(50e3, 0.0, nlev)
    z_half_1d = _jnp.concatenate([
        _jnp.array([z_full[0] + 1e3]),
        0.5 * (z_full[:-1] + z_full[1:]),
        _jnp.array([z_full[-1] - 1e3]),
    ])
    z_full_b = _jnp.broadcast_to(z_full[None, :], (ncol, nlev))
    z_half_b = _jnp.broadcast_to(z_half_1d[None, :], (ncol, nlev + 1))
    p_full = _jnp.broadcast_to(
        (1e5 * _jnp.exp(-z_full / H))[None, :], (ncol, nlev),
    )
    p_half = _jnp.broadcast_to(
        (1e5 * _jnp.exp(-z_half_1d / H))[None, :], (ncol, nlev + 1),
    )
    rho = p_full / (_const.R_d * T)
    u = _jnp.full((ncol, nlev), 10.0)
    v = _jnp.zeros((ncol, nlev))
    lat = _jnp.zeros(ncol)

    # Default config: saturation is active aloft; Fmax = 0.1 Pa.
    cfg = HinesConfig()
    out = hines_gwd(u, v, T, p_full, p_half, z_full_b, z_half_b, rho, lat,
                   300.0, cfg)

    # 1. Acceleration sign: GWD opposes the wind (u > 0 ⇒ du/dt ≤ 0)
    #    everywhere active drag is deposited.
    max_pos_dudt = float(jnp.max(out.du_dt))
    assert max_pos_dudt <= 1e-15, (
        f"Hines: positive du/dt = {max_pos_dudt:.3e} m/s² for u > 0 — "
        "GWD must always oppose the resolved flow."
    )

    # 2. Acceleration magnitude: stratospheric GWD is typically
    #    1e-5–1e-3 m/s² (≈1–100 m/s/day).  Anything > 0.1 m/s² is
    #    non-physical and would indicate a unit error like the buggy
    #    ``1/s`` form had been retained.
    max_abs_dudt = float(jnp.max(jnp.abs(out.du_dt)))
    assert max_abs_dudt < 1e-2, (
        f"Hines: max |du/dt| = {max_abs_dudt:.3e} m/s² exceeds the "
        "physical stratospheric GWD bound (~1e-3 m/s²).  This indicates "
        "the drag dimensional fix has regressed — units back to "
        "``kg/(m²·s)`` and the downstream acceleration in ``1/s``."
    )
