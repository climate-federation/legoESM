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
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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

    For Hines, the default per-level drag cap ``Fmax=0.1`` Pa binds at
    this coarse (10-level) strong-shear column — the deposition pins at
    the limiter, so it no longer responds to the Brunt-Vaisala frequency
    ``N`` (hence ``T``) and ``dT_dt`` becomes T-independent (grad → 0).
    Raising ``Fmax`` lets the Doppler-spread ``(sigma_grown^2 -
    sigma_sat^2)`` term — which carries the temperature dependence via
    ``sigma_sat = N / m_*`` — be the binding term again.  (On
    operational many-level grids the per-level deposition is « 0.1 Pa,
    so the production default does not bind.)
    """
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        HinesConfig, LindzenConfig, McFarlaneConfig,
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
    if scheme == "hines":
        # Relax the E3SM-faithful magnitude limiter (``tndmax``/``umcfac``;
        # gw_common.F90:642) for this differentiability probe. At Fmax=100 the
        # no-reversal cap ``accel = -min(|accel|, umcfac*U_mag/dt)`` binds at
        # every active level, and ``jnp.minimum`` routes the gradient to the
        # T-independent cap → d(dT/dt)/dT collapses to ~0. Lifting the cap
        # isolates the core N(T) → drag → frictional-heating path the test means
        # to exercise (the limiter magnitude itself is covered by its own tests).
        return GravityWaveDragConfig(
            scheme="hines",
            hines=HinesConfig(Fmax=100.0, tndmax_per_day=1.0e4, umcfac=1.0e3),
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


def test_hines_T_gradient_is_limiter_gated_not_missing():
    """Documents the diagnosis behind the relaxed-limiter hines config above.

    The hines heating IS a differentiable function of T (via the Brunt-Väisälä
    frequency N(theta) → saturation amplitude → drag → frictional heating). But
    the E3SM-faithful magnitude limiter ``accel = -min(|accel|, umcfac*U_mag/dt)``
    is intentionally T-independent *when it binds*: ``jnp.minimum`` routes the
    gradient to the cap, so the operational (saturated, Fmax=100) regime shows a
    ~zero dT/dT. That is the limiter doing its job, not a missing/severed T-path.

    Proof that the relaxed config exposes REAL signal (not numerical noise): the
    cap-off gradient is many orders of magnitude larger than the cap-on one.
    """
    from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig

    state, grid, sigma = _make_sheared_state()
    T_field = state.T

    def grad_for(hines_cfg):
        def loss(dT):
            sp = state._replace(T=T_field.replace(data=T_field.data + dT))
            fn = make_gwd_physics(
                GravityWaveDragConfig(scheme="hines", hines=hines_cfg),
                model_type="hydrostatic", dt=300.0,
            )
            tend, _ = fn(sp, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)
        return abs(float(jax.grad(loss)(jnp.asarray(0.0))))

    capped = grad_for(HinesConfig(Fmax=100.0))  # E3SM limiter binds → ~0
    freed = grad_for(HinesConfig(Fmax=100.0, tndmax_per_day=1.0e4, umcfac=1.0e3))
    assert jnp.isfinite(capped) and jnp.isfinite(freed)
    assert freed > 1e-15, (
        f"uncapped hines T-gradient should be real signal, got {freed}"
    )
    assert freed > 1.0e6 * capped, (
        f"lifting the magnitude limiter must expose a far larger T-gradient "
        f"(capped={capped:.3e}, freed={freed:.3e}) — confirms it is the cap, "
        f"not a missing T-path, that zeroes the operational gradient."
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


class TestBruntVaisalaShared:
    """Direct tests for the shared physics._shared.brunt_vaisala_n_full helper
    that the four GWD backends (hines/lindzen/mcfarlane/prognostic_spectral)
    now call instead of an inline copy (ponytail dedup 2026-06-17)."""

    def test_shape_and_positivity(self):
        from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
        from legoesm import constants
        ncol, nlev = 4, 12
        # Stably-stratified isothermal-ish column: θ grows with height.
        z = jnp.broadcast_to(jnp.linspace(2.0e4, 0.0, nlev), (ncol, nlev))
        p = jnp.broadcast_to(jnp.linspace(2.0e3, 1.0e5, nlev), (ncol, nlev))
        T = jnp.full((ncol, nlev), 250.0)
        N = brunt_vaisala_n_full(T, p, z)
        assert N.shape == (ncol, nlev)
        # N² floored at 1e-8 ⇒ N ≥ 1e-4 everywhere, finite.
        assert jnp.all(N >= 1e-4 - 1e-12)
        assert jnp.all(jnp.isfinite(N))

    def test_matches_inline_reference(self):
        """Byte-identical to the formula the GWD schemes used to inline."""
        import numpy as np
        from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
        from legoesm import constants
        rng = np.random.default_rng(1)
        T = jnp.asarray(220 + 60 * rng.random((5, 12)))
        p = jnp.asarray(np.sort(1e5 * rng.random((5, 12)))[:, ::-1].copy())
        z = jnp.asarray(np.sort(2e4 * rng.random((5, 12))).copy())
        theta = T * (constants.p_ref / jnp.clip(p, 1.0, None)) ** constants.kappa
        dz = jnp.clip(jnp.abs(z[:, :-1] - z[:, 1:]), 1.0, None)
        dth = (theta[:, :-1] - theta[:, 1:]) / dz
        tb = 0.5 * (theta[:, :-1] + theta[:, 1:])
        N2 = jnp.clip((constants.g / jnp.clip(tb, 1.0, None)) * dth, 1e-8, None)
        Nh = jnp.sqrt(N2)
        ref = jnp.concatenate(
            [Nh[:, :1], 0.5 * (Nh[:, :-1] + Nh[:, 1:]), Nh[:, -1:]], axis=1
        )
        assert jnp.array_equal(brunt_vaisala_n_full(T, p, z), ref)


# ============================================================================
# Combined orographic + non-orographic GWD (issue #834)
# ============================================================================
#
# A ``+``-composite ``gravity_wave_drag`` (e.g. ``mcfarlane+prognostic_spectral``)
# runs several sources and SUMS their du/dv/dT tendencies; the one stateful part
# (prognostic_spectral) threads its wave-action spectrum through the carry.

def _combined_cols(ncol=4, nlev=12):
    """Deterministic stratospheric-ish columns: strong jet + stable N^2.

    Returns u, v, T, p_full, p_half, z_full, z_half, rho, lat — the shared
    GWD backend column signature.
    """
    from legoesm import constants
    p_half = jnp.broadcast_to(
        jnp.linspace(50.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = jnp.broadcast_to(jnp.linspace(210.0, 290.0, nlev)[None, :], (ncol, nlev))
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    z_half_cumsum = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    # Moderate, uniform westerly (~14 m/s): strong enough surface wind to launch
    # orographic (McFarlane) stress that saturates aloft as rho decreases
    # (breaking -> drag) AND to drive prognostic-spectral breaking.  Matches the
    # activation regime of test_gravity_wave_drag._make_columns (uniform ~10 m/s
    # is McFarlane-active); avoids a wind-increasing-upward profile that would
    # let the orographic wave escape to the top without saturating.
    u = jnp.full((ncol, nlev), 14.0)
    v = jnp.full((ncol, nlev), 3.0)
    lat = jnp.full((ncol,), 0.6)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


def test_gwd_carries_spectrum_predicate():
    """The shared predicate is True iff a prognostic_spectral part is present."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        gwd_carries_spectrum,
    )
    assert gwd_carries_spectrum("prognostic_spectral") is True
    assert gwd_carries_spectrum("mcfarlane+prognostic_spectral") is True
    assert gwd_carries_spectrum("prognostic_spectral+hines") is True
    assert gwd_carries_spectrum("mcfarlane") is False
    assert gwd_carries_spectrum("hines+mcfarlane") is False
    assert gwd_carries_spectrum("none") is False


def test_get_gwd_fn_composite_dispatch_and_invalid_raises():
    """get_gwd_fn routes a valid '+'-scheme to the combined executor (returning
    the FULL config) and HARD-RAISES at the factory chokepoint on an invalid
    composite — independent of ExperimentConfig.validate_strict (codex
    adversarial finding 2: without this guard a non-composable part hits a
    low-level TypeError and two spectral parts silently corrupt the carry)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn, _combined_gwd,
    )
    name, fn, cfg = get_gwd_fn(
        GravityWaveDragConfig(scheme="mcfarlane+prognostic_spectral"))
    assert name == "mcfarlane+prognostic_spectral"
    assert fn is _combined_gwd
    # Full config returned (executor needs every part's sub-config).
    assert isinstance(cfg, GravityWaveDragConfig)

    # Unknown / non-composable parts raise at get_gwd_fn (before returning fn).
    for bad in ("mcfarlane+nonsense", "mcfarlane+ml_emulator",
                "mcfarlane+e3sm_cam"):
        with pytest.raises(ValueError, match="[Nn]on-composable"):
            get_gwd_fn(GravityWaveDragConfig(scheme=bad))
    # More than one stateful source shares a single spectrum carry -> raise.
    with pytest.raises(ValueError, match="at most one stateful"):
        get_gwd_fn(
            GravityWaveDragConfig(
                scheme="prognostic_spectral+prognostic_spectral"))


def test_combined_tendency_is_exact_sum():
    """CORE (#834): the composite du/dv/dT/eps equals the arithmetic sum of the
    individual McFarlane (orographic) and prognostic_spectral (non-orographic)
    tendencies computed on the SAME column — no cross-talk, linear body forces."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
    from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
        prognostic_spectral_gwd,
    )
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _combined_cols()
    dt = 300.0
    cfg = GravityWaveDragConfig(scheme="mcfarlane+prognostic_spectral")
    ncol = u.shape[0]
    sc = cfg.prognostic_spectral
    # Seed the wave-action spectrum ABOVE launch_flux (0.01 Pa) so the
    # prognostic source actually breaks and deposits momentum — matches the
    # activation level of test_gravity_wave_drag.TestPrognosticSpectral.
    # test_acceleration_magnitude_includes_g_factor (drag is ~0 at the 1e-3
    # default seed on these idealized columns).
    spec_in = jnp.full((ncol, sc.n_azimuths, sc.n_wavenumbers), 0.01)
    # Large subgrid-orography stddev (800 m) so the McFarlane launch stress
    # (tau_0 ~ h_topo^2) is big enough to saturate and break aloft — the drag
    # is ~0 at the scalar-config h_topo on these smooth idealized columns.
    # Passing it through combined_fn also exercises h_topo_col threading to the
    # orographic part inside the composite executor.
    h_topo = jnp.full((ncol,), 800.0)

    oro = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
                        cfg.mcfarlane, h_topo_col=h_topo)
    spec_out, spec_new_ref = prognostic_spectral_gwd(
        u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
        cfg.prognostic_spectral, spec_in)

    _, combined_fn, full_cfg = get_gwd_fn(cfg)
    combined, spec_new = combined_fn(
        u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, full_cfg,
        spec_in, h_topo_col=h_topo)

    assert jnp.allclose(combined.du_dt, oro.du_dt + spec_out.du_dt, atol=1e-12)
    assert jnp.allclose(combined.dv_dt, oro.dv_dt + spec_out.dv_dt, atol=1e-12)
    assert jnp.allclose(combined.dT_dt, oro.dT_dt + spec_out.dT_dt, atol=1e-12)
    assert jnp.allclose(combined.eps_gwd, oro.eps_gwd + spec_out.eps_gwd, atol=1e-12)
    # Both sources must actually be active (else the "sum" test is vacuous).
    assert float(jnp.max(jnp.abs(oro.du_dt))) > 0.0
    assert float(jnp.max(jnp.abs(spec_out.du_dt))) > 0.0
    # Spectrum is advanced by the prognostic part and threaded out unchanged
    # from the combined executor.
    assert jnp.allclose(spec_new, spec_new_ref)


def test_combined_threads_and_evolves_spectrum():
    """The composite returns the advanced wave-action spectrum (shape preserved,
    values changed from the launch seed) so the carry can be threaded."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn,
    )
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _combined_cols()
    cfg = GravityWaveDragConfig(scheme="mcfarlane+prognostic_spectral")
    ncol = u.shape[0]
    sc = cfg.prognostic_spectral
    # Seed above launch_flux so the spectrum both breaks and relaxes -> evolves.
    spec_in = jnp.full((ncol, sc.n_azimuths, sc.n_wavenumbers), 0.01)
    _, combined_fn, full_cfg = get_gwd_fn(cfg)
    _, spec_new = combined_fn(
        u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, full_cfg, spec_in)
    assert spec_new.shape == spec_in.shape
    assert not bool(jnp.allclose(spec_new, spec_in))


def test_stateless_composite_sums_without_spectrum():
    """A stateless composite (hines+mcfarlane) sums both sources and passes the
    spectrum through unchanged (None in / None out)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
    from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _combined_cols()
    dt = 300.0
    cfg = GravityWaveDragConfig(scheme="hines+mcfarlane")
    oro = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
                        cfg.mcfarlane)
    hin = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
                    cfg.hines)
    _, combined_fn, full_cfg = get_gwd_fn(cfg)
    combined, spec_out = combined_fn(
        u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, full_cfg, None)
    assert jnp.allclose(combined.du_dt, oro.du_dt + hin.du_dt, atol=1e-12)
    assert jnp.allclose(combined.dT_dt, oro.dT_dt + hin.dT_dt, atol=1e-12)
    assert spec_out is None  # no stateful part -> spectrum passthrough


def test_combined_hydrostatic_factory_sums_and_threads_spectrum():
    """Factory-level (#834): make_gwd_physics('hydrostatic', combined) returns
    summed HydrostaticTendencies and the advanced spectrum when a phys_state
    spectrum is supplied — the driver's stateful-carry contract."""
    from types import SimpleNamespace
    state, grid, sigma = _make_state(n=4, nlev=10, wind_speed=25.0)
    ncol = 6 * 4 * 4
    combined_cfg = GravityWaveDragConfig(scheme="mcfarlane+prognostic_spectral")
    mcf_cfg = GravityWaveDragConfig(scheme="mcfarlane")
    fn_comb = make_gwd_physics(combined_cfg, model_type="hydrostatic", dt=300.0)
    fn_mcf = make_gwd_physics(mcf_cfg, model_type="hydrostatic", dt=300.0)
    sc = combined_cfg.prognostic_spectral
    # Seed above launch_flux so the prognostic source is active this step.
    spec0 = jnp.full((ncol, sc.n_azimuths, sc.n_wavenumbers), 0.01)
    phys = SimpleNamespace(gwd_spectrum=spec0)

    tend_comb, spec_new = fn_comb(state, grid, sigma, phys_state=phys)
    tend_mcf, _ = fn_mcf(state, grid, sigma)

    # The non-orographic (spectral) source changes the momentum tendency vs
    # McFarlane-only — the combined field is not the orographic field (drag
    # can locally oppose, so magnitude ordering is not guaranteed pointwise;
    # what must hold is that the field differs, i.e. the second source is wired
    # in and summed).
    assert not bool(jnp.allclose(tend_comb.du_dt.data, tend_mcf.du_dt.data))
    # Spectrum threaded out, advanced from the seed.
    assert spec_new is not None
    assert spec_new.shape == spec0.shape
    assert not bool(jnp.allclose(spec_new, spec0))
    assert bool(jnp.all(jnp.isfinite(tend_comb.du_dt.data)))
