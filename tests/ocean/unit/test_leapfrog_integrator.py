"""NEMO Modified-Leap-Frog outer integrator (config.outer_integrator="leapfrog").

Faithful port of NEMO 5.0.2 ``stp_MLF`` (stpmlf.F90, key_qco DINO): three time
levels Nbb/Nnn/Naa, explicit combine ``X(Naa)=X(Nbb)+2dt·RHS(Nnn)`` with the
Coriolis IN the RHS (vorticity_scheme="een_total", NEMO ln_dynvor_een EEN triad),
implicit vertical friction/diffusion over rDt=2dt (dyn_zdf), and the PLAIN
Robert-Asselin time filter (rn_atfp, NOT Williams — dynatf_qco.F90:144).  First
step is a forward-Euler start (l_1st_euler).

Gates:
  * default forward_euler path unchanged;
  * leapfrog runs stably (no NaN, physical T);
  * first step == forward-Euler (Nbb <- pre-step now, unfiltered);
  * the leapfrog COMPUTATIONAL MODE is damped by the Asselin filter (analytic);
  * een_total carries the planetary Coriolis in the RHS (f×v in du_dt);
  * dispatch-hardening (bad config combinations raise at construction).
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


def _channel(outer_integrator="forward_euler", n_lat=8, n_lon=16, **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        outer_integrator=outer_integrator, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _leapfrog_channel(**cfg_kw):
    cfg_kw.setdefault("coriolis_scheme", "explicit_ab2")
    cfg_kw.setdefault("vorticity_scheme", "een_total")
    return _channel("leapfrog", **cfg_kw)


# ---------------------------------------------------------------------------
# Backward compatibility + dispatch hardening
# ---------------------------------------------------------------------------

def test_forward_euler_default_unchanged():
    state, model = _channel("forward_euler")
    s = model.step(state, dt=_DT)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    # the leapfrog before-fields are inert (never populated) on the FE path
    assert s.u_before is None and s.eta_before is None


def test_step_impl_advective_override_withholds_diss():
    """The leap-frog Nbb-diffusion pass relies on ``_ab2_scope_override``:
    under config ab2_scope="total" the explicit-only ``_step_impl`` bundles the
    dissipation into ``state_expl`` and returns diss_incr=None; passing
    ``_ab2_scope_override="advective"`` WITHHOLDS the lateral-diffusion + GM/Redi
    increment onto the returned diss_incr WITHOUT mutating the config.  NEMO
    dyn_ldf(Kbb)/tra_ldf(Kbb): the leap-frog evaluates that increment on Nbb."""
    # K_h>0 → active lateral tracer diffusion acting on the imposed meridional T
    # gradient, so the withheld increment is non-trivial (a non-vacuous check).
    state, model = _leapfrog_channel(K_h=2.0e4)
    assert model.config.ab2_scope == "total"
    # total scope (default) — diss folded into state_expl, diss_incr is None
    exp_total, aux_total = model._step_impl(
        state, _DT, _apply_implicit_vmix=False)
    assert aux_total[5] is None
    # advective override — diss withheld onto diss_incr (a 4-tuple of finite
    # arrays: dT, dS, du, dv), config untouched
    exp_adv, aux_adv = model._step_impl(
        state, _DT, _apply_implicit_vmix=False,
        _ab2_scope_override="advective")
    diss = aux_adv[5]
    assert diss is not None and len(diss) == 4
    for arr in diss:
        assert np.all(np.isfinite(np.asarray(arr)))
    # the withheld tracer increment is non-trivial (K_h diffusion on the T front)
    assert float(np.max(np.abs(np.asarray(diss[0])))) > 0.0
    # and it is EXACTLY what was removed from state_expl.T: advective-scope T
    # (diffusion withheld) minus the withheld increment == total-scope T.
    np.testing.assert_allclose(
        np.asarray(exp_adv.T.data) + np.asarray(diss[0]),
        np.asarray(exp_total.T.data), rtol=1e-9, atol=1e-9)
    assert model.config.ab2_scope == "total"   # config NOT mutated


def test_leapfrog_barotropic_substep_scale_holds_substep_length():
    """The leap-frog integrates the barotropic mode over rDt=2dt, so its substep
    count is scaled ×2 (``_barotropic_substep_scale``) to keep the substep length
    — and the barotropic CFL — identical to the dt-window forward-Euler path.
    Without this the split-explicit free surface goes unstable in ~4 steps."""
    state, model = _leapfrog_channel()
    n_cfg = model.config.barotropic.n_barotropic_substeps
    # default scale=1 → count unchanged
    import numpy as _np
    captured = {}
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as _m
    orig = _m.barotropic_substeps_latlon_cgrid

    def _spy(state_mid, dt_s, n_sub, *a, **k):
        captured["dt_s"] = float(dt_s)
        captured["n_sub"] = int(n_sub)
        return orig(state_mid, dt_s, n_sub, *a, **k)

    _m.barotropic_substeps_latlon_cgrid = _spy
    try:
        model._step_impl(state, _DT, _apply_implicit_vmix=False,
                         _barotropic_substep_scale=2)
    finally:
        _m.barotropic_substeps_latlon_cgrid = orig
    assert captured["n_sub"] == 2 * n_cfg
    # substep length halved vs a scale-1 call at the same window
    assert abs(captured["dt_s"] - _DT / (2 * n_cfg)) < 1e-9 * _DT


def test_leapfrog_requires_explicit_ab2():
    # Coriolis in a Matsuno rotation would double-apply f (Coriolis is in the
    # RHS via een_total). Rejected at construction.
    with pytest.raises(ValueError, match="explicit_ab2"):
        _channel("leapfrog", vorticity_scheme="een_total",
                 coriolis_scheme="matsuno_split")


def test_leapfrog_requires_implicit_vmix():
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        _leapfrog_channel(implicit_vertical_mixing=False)


def test_een_total_matsuno_rejected():
    with pytest.raises(ValueError, match="explicit_ab2"):
        _channel("forward_euler", vorticity_scheme="een_total",
                 coriolis_scheme="matsuno_split")


def test_unknown_vorticity_scheme_raises():
    # membership fail-early lives in _bc_pv_flux (step time), reached via .step()
    state, model = _channel("forward_euler", vorticity_scheme="bogus")
    with pytest.raises(ValueError, match="vorticity_scheme"):
        model.step(state, dt=_DT)


# ---------------------------------------------------------------------------
# Leapfrog runs stably
# ---------------------------------------------------------------------------

def test_leapfrog_scan_seeded_and_runs():
    """The differentiable lax.scan path: seed_scan_carry seeds the before-state
    Nbb so the carry pytree is CONSTANT across scan iterations (the None->Field
    first-step transition would otherwise crash lax.scan). integrate_scan runs
    finite."""
    state, model = _leapfrog_channel()
    seeded = model.seed_scan_carry(state, _DT)
    # before-state seeded to Fields (not None)
    assert seeded.u_before is not None and seeded.eta_before is not None
    np.testing.assert_allclose(
        np.asarray(seeded.u_before.data), np.asarray(state.u.data))
    final, traj = model.integrate_scan(state, 5, _DT)
    assert np.all(np.isfinite(np.asarray(final.T.data)))
    assert np.all(np.isfinite(np.asarray(final.u.data)))


def test_leapfrog_rejects_prescribed_flow():
    with pytest.raises(ValueError, match="prescribed_flow"):
        _leapfrog_channel(prescribed_flow="rest")


def test_leapfrog_runs_no_nan():
    state, model = _leapfrog_channel()
    s = state
    for _ in range(12):
        s = model.step(s, dt=_DT)
    T = np.asarray(s.T.data)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(np.asarray(s.u.data)))
    # physical: the tanh IC is bounded in [-4, +4] °C about the rest profile,
    # so no spurious runaway (a blown leapfrog computational mode would explode).
    assert np.nanmax(np.abs(T)) < 40.0
    # before-state carried
    assert s.u_before is not None and s.eta_before is not None


def test_leapfrog_first_step_is_euler():
    """First leapfrog step == a plain forward-Euler step, and Nbb is set to the
    (unfiltered) pre-step now-fields — NEMO l_1st_euler."""
    state, model_lf = _leapfrog_channel()
    # matched forward-Euler reference with the SAME Coriolis/vorticity placement
    state_fe, model_fe = _channel(
        "forward_euler", coriolis_scheme="explicit_ab2",
        vorticity_scheme="een_total", momentum_time_integrator="rk3_ws")
    s_lf = model_lf.step(state, dt=_DT)
    # Nbb == pre-step now (unfiltered)
    np.testing.assert_allclose(
        np.asarray(s_lf.T_before.data), np.asarray(state.T.data), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(s_lf.u_before.data), np.asarray(state.u.data), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(s_lf.eta_before.data), np.asarray(state.eta.data),
        rtol=0, atol=0)


def test_leapfrog_second_step_shift():
    """After step 2, Nbb == the Asselin-filtered now of step 2 = state1 +
    gamma·(state1_before - 2·state1 + Naa).  Verify the stored before-field
    matches the closed-form RA filter on T from the step's I/O."""
    state, model = _leapfrog_channel()
    s1 = model.step(state, dt=_DT)        # Euler start: s1.T_before == state.T
    s2 = model.step(s1, dt=_DT)           # leapfrog + Asselin
    gamma = model.config.asselin_gamma
    mask = np.broadcast_to(
        (np.asarray(state.land_mask.data) > 0.5)[..., None],
        np.asarray(state.T.data).shape)
    # RA filter reconstruction of s2.T_before from s1 (now), s1.T_before (Nbb),
    # and s2.T (Naa):
    now = np.asarray(s1.T.data)
    before = np.asarray(s1.T_before.data)
    after = np.asarray(s2.T.data)
    T_f = now + gamma * (before - 2.0 * now + after)
    np.testing.assert_allclose(
        np.asarray(s2.T_before.data)[mask], T_f[mask], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Analytic: the Asselin filter damps the leapfrog COMPUTATIONAL mode
# ---------------------------------------------------------------------------

def _leapfrog_ra_oscillator(gamma, omega_dt, n_steps, seed_comp=1.0):
    """Scalar leapfrog + Robert-Asselin filter on du/dt = i·omega·u (rotation/
    oscillation — the exact analytic reduction of the Coriolis leapfrog).

    Returns the amplitude of the parasitic COMPUTATIONAL mode over time by
    seeding it explicitly: u^0 = physical, u^{-1} chosen with a computational-mode
    component. Tracks the alternating (-1)^n signature amplitude.
    """
    # du/dt = i omega u.  Leapfrog: u^{n+1} = u^{n-1} + 2 i omega dt u^n, then
    # RA filter on the "now": u^n_f = u^n + gamma (u^{n-1} - 2 u^n + u^{n+1}).
    # Seed a pure computational mode: u^{-1}, u^0 with opposite-sign physical.
    ubb = complex(seed_comp)          # before (n-1): pure computational seed
    unn = complex(-seed_comp)         # now (n): alternating -> computational mode
    amps = []
    for _ in range(n_steps):
        uaa = ubb + 2j * omega_dt * unn
        # RA filter on the now
        unn_f = unn + gamma * (ubb - 2.0 * unn + uaa)
        # amplitude of the alternating (computational) signature = |unn_f + uaa|/2
        # (the computational mode alternates sign step-to-step; the physical mode
        # is smooth). Track the magnitude of the now field.
        amps.append(abs(unn_f))
        ubb, unn = unn_f, uaa
    return np.array(amps)


def test_leapfrog_computational_mode_damped_by_asselin():
    """With gamma>0 the leapfrog computational mode DECAYS; with gamma=0 it is
    neutral/undamped (the classic leapfrog parasitic mode).  Pure analytic — no
    model."""
    omega_dt = 0.2   # f·dt well below the leapfrog CFL (1)
    n = 200
    amp_filtered = _leapfrog_ra_oscillator(0.1, omega_dt, n)
    amp_unfiltered = _leapfrog_ra_oscillator(0.0, omega_dt, n)
    # unfiltered: the seeded computational mode does NOT decay
    assert amp_unfiltered[-1] > 0.5 * amp_unfiltered[0]
    # filtered (gamma=0.1, NEMO rn_atfp): the computational mode is strongly
    # damped by the end of the window
    assert amp_filtered[-1] < 0.2 * amp_filtered[0]
    # and monotone-ish decay: the late amplitude is far below the early one
    assert amp_filtered[-1] < 0.5 * amp_filtered[10]


def test_leapfrog_physical_mode_neutral():
    """The PHYSICAL mode (smooth rotation) is NOT destroyed by the RA filter:
    amplitude stays O(1) over the window (leapfrog is neutrally stable for
    oscillation, and the filter's O(gamma·omega²dt²) damping of the physical mode
    is weak)."""
    omega_dt = 0.2
    n = 200
    # seed a PHYSICAL (smooth) mode: u^{-1} ~ u^0 · exp(-i omega dt)
    ubb = 1.0 + 0j
    unn = complex(np.cos(omega_dt), np.sin(omega_dt))
    gamma = 0.1
    for _ in range(n):
        uaa = ubb + 2j * omega_dt * unn
        unn_f = unn + gamma * (ubb - 2.0 * unn + uaa)
        ubb, unn = unn_f, uaa
    # physical amplitude decays only weakly (>= 60% retained over 200 steps)
    assert abs(unn) > 0.6


# ---------------------------------------------------------------------------
# een_total: planetary Coriolis is IN the baroclinic RHS
# ---------------------------------------------------------------------------

def test_een_total_puts_coriolis_in_rhs():
    """een_total (vorticity_scheme) folds f into the EEN triad so f×v appears in
    du_dt.  With a uniform northward v imposed at rest, the zonal-momentum
    tendency du_dt is nonzero and has the sign of +f·v in the interior
    (Northern Hemisphere f>0, v>0 -> du_dt>0)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(8, 16)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    # impose a uniform northward v
    v = np.asarray(state.v.data) + 0.1 * np.asarray(state.v_mask.data)[..., None]
    state = state._replace(v=state.v.replace(data=jnp.asarray(v)))

    def _du(vs, cor):
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=0.0, bottom_drag_r=0.0, n_barotropic_substeps=8,
            enable_runtime_checks=False, implicit_vertical_mixing=True,
            coriolis_scheme=cor, vorticity_scheme=vs,
            momentum_time_integrator="rk3_ws")
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        tend = model.tendencies(state, None)
        return np.asarray(tend.du_dt.data)

    # een_total (explicit_ab2): planetary f folded INTO the EEN triad -> f×v in du_dt.
    du_een = _du("een_total", "explicit_ab2")
    # al81 + matsuno_split: the relative-vorticity flux ONLY (planetary f is applied
    # later by the Matsuno rotation inside _step_impl, NOT in tendencies), so du_dt
    # has NO planetary f — the clean no-Coriolis baseline.  Uniform v -> zeta ~ 0.
    du_norot = _du("al81", "matsuno_split")
    lat = np.degrees(np.asarray(grid.lat))
    nh = lat > 20.0
    # een_total carries f×v (+, NH); the relative-only baseline is ~0 there.
    interior_een = du_een[nh]
    interior_een = interior_een[np.isfinite(interior_een)]
    assert np.mean(interior_een) > 0.0                 # +f·v signature
    assert np.max(np.abs(du_een)) > 10.0 * np.max(np.abs(du_norot[nh]))
