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


def test_live_split_under_een_total_requires_een_stencil():
    # LIVE barotropic-Coriolis split (node 16) under een_total: the pre-step
    # subtraction must use the SAME EEN stencil the substep applies live, else
    # the 4-pt-avg subtraction leaves an O(1) residual Coriolis. avg => reject.
    with pytest.raises(ValueError, match="barotropic_coriolis"):
        _leapfrog_channel(barotropic_coriolis="avg",
                          barotropic_coriolis_split="live")


def test_live_een_barotropic_coriolis_runs_no_nan():
    # The LIVE EEN barotropic Coriolis under the leapfrog builds, validates, and
    # steps finite (no double-count blow-up from the frozen+live pairing).
    state, model = _leapfrog_channel(barotropic_coriolis="een",
                                     barotropic_coriolis_split="live")
    assert model.config.barotropic_coriolis_split == "live"
    s = state
    for _ in range(4):
        s = model.step(s, dt=_DT)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert np.all(np.isfinite(np.asarray(s.eta.data)))


def test_boxcar_ab3_live_split_runs_no_nan():
    # NEMO nn_bt_flt=2 (barotropic_time_filter="nemo_boxcar_ab3") = the AB3
    # velocity predictor + ts_bck_interp(alpha=0) ssh temporal dissipation +
    # boxcar averaging, composed WITH the live EEN barotropic Coriolis (the
    # nemo_dino_kamm_mlf config). The validator must ACCEPT this pairing (only
    # nemo_ab3am4 is rejected with the live split), and it must step finite.
    state, model = _leapfrog_channel(
        barotropic_coriolis="een", barotropic_coriolis_split="live",
        barotropic_time_filter="nemo_boxcar_ab3")
    assert model.config.barotropic.barotropic_time_filter == "nemo_boxcar_ab3"
    s = state
    for _ in range(6):
        s = model.step(s, dt=_DT)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert np.all(np.isfinite(np.asarray(s.eta.data)))


def test_leapfrog_before_seed_wide_halo_not_implemented():
    # The MLF before-level barotropic seed is not wired into the wide-halo path;
    # the leap-frog step must raise NotImplementedError rather than silently drop
    # the Nbb seed (barotropic_wide_halo=True + a non-ab3 filter reaches the
    # before-state guard). First step is a forward-Euler start (no seed), so
    # advance one step to populate the before-fields, then the seeded pass fires.
    state, model = _leapfrog_channel(
        barotropic_time_filter="nemo_boxcar_centred",
        barotropic_wide_halo=True, barotropic_local_subcycle_clamp=True)
    s = model.step(state, dt=_DT)   # forward-Euler start (no before-seed yet)
    with pytest.raises(NotImplementedError, match="wide-halo"):
        model.step(s, dt=_DT)       # leapfrog pass → before-seed → guard


def test_ab3am4_live_split_still_rejected():
    # nemo_ab3am4 (nn_bt_flt=3, cross-window carry => substep-0 extrapolates)
    # stays incompatible with the live split; only the ramp-every-step
    # nemo_boxcar_ab3 (nn_bt_flt=2) is allowed.
    with pytest.raises(ValueError, match="nemo_ab3am4"):
        _leapfrog_channel(barotropic_coriolis="een",
                          barotropic_coriolis_split="live",
                          barotropic_time_filter="nemo_ab3am4")


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


def test_leapfrog_from_rest_nemo_before_and_burchard_do_not_crash():
    """#1317 regression: a FRESH (unbridged, unseeded) from-rest state on the
    nemo_dino_kamm_mlf combo (tke_n2_time_level="nemo_before" +
    tke_shear_production="nemo_burchard") used to raise/AttributeError on
    step 0 -- ``_leapfrog_step``'s Euler-start branch called ``_step_impl``
    BEFORE seeding ``state.T_before``/``u_before``/``v_before``, and both
    ``_n2_nemo_before_tracers`` and the Burchard-shear guard in
    ``k_profiles.py`` read those fields unconditionally. The fix seeds a
    LOCAL before:=now copy for that first ``_step_impl`` call only --
    NEMO's own cold-start convention (istate.F90: ``ts(:,:,:,:,Kmm) =
    ts(:,:,:,:,Kbb)`` before stp_MLF is ever entered, so Nbb==Nnn on the
    first step). A from-rest run must now construct AND step at least twice
    with no SystemExit/ValueError/NaN."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig, VerticalMixingConfig,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    _lateral_mixing_none = type(OceanPhysicsConfig().lateral_mixing)(
        scheme="none")
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(tke_n2_time_level="nemo_before",
                          tke_shear_production="nemo_burchard")),
        convection=OceanConvectionConfig(scheme="none"),
        lateral_mixing=_lateral_mixing_none,
    )
    state, model = _leapfrog_channel(
        physics=physics, barotropic_forcing_centred=True,
        barotropic_een_seed="nemo_kmm")
    # Fresh from-rest state: no bridge, no before-fields populated.
    assert state.T_before is None and state.u_before is None

    s1 = model.step(state, dt=_DT)   # step 0: the crash site pre-fix
    assert np.all(np.isfinite(np.asarray(s1.T.data)))
    assert np.all(np.isfinite(np.asarray(s1.u.data)))
    s2 = model.step(s1, dt=_DT)      # step 1: genuine leapfrog + Asselin
    assert np.all(np.isfinite(np.asarray(s2.T.data)))
    assert np.all(np.isfinite(np.asarray(s2.u.data)))


def test_leapfrog_from_rest_nemo_face_native_does_not_crash():
    """#1226 sh2_walk.py Candidate E/F regression: nemo_dino_kamm_mlf's
    ACTUAL runtime combination -- tke_shear_production="nemo_face_native"
    together with bottom_tke_bc=True (the real card also sets
    tke_bottom_bc=True) on a PARTIAL-CELL z-coordinate (the card always
    uses one; nemo_face_native requires z_coord.is_active). Two real bugs
    were caught and fixed by this exact combination during development:
    (1) the model's cc_state pre-collapse silently defeated the face-
    native geometry (fixed: ocean_model_latlon_cgrid.py's fallback K-profile
    call now keeps u/v at their raw C-grid face shape for this option), and
    (2) _tke_bottom_dirichlet's take_along_axis assumed cell-centred u/v
    (fixed: it now collapses locally when needed). A from-rest run must
    step at least twice with no crash/NaN."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig, VerticalMixingConfig,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    n_lat, n_lon, n_levels, H_max = 8, 16, 5, 3000.0
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z0c = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    H_bathy = jnp.full((n_lat, n_lon), H_max * 0.62)
    z = create_partial_cell_coordinate(z0c, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    assert state.T_before is None and state.u_before is None

    _lateral_mixing_none = type(OceanPhysicsConfig().lateral_mixing)(
        scheme="none")
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(tke_n2_time_level="nemo_before",
                          tke_shear_production="nemo_face_native",
                          bottom_tke_bc=True, prognostic=True)),
        convection=OceanConvectionConfig(scheme="none"),
        lateral_mixing=_lateral_mixing_none,
    )
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4,
        bottom_drag_r=1.0e-3, bottom_drag_scheme="nemo_quadratic",
        bottom_drag_cd0=1.0e-3, bottom_drag_cdmax=0.1,
        bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        implicit_vertical_mixing=True,
        outer_integrator="leapfrog", coriolis_scheme="explicit_ab2",
        vorticity_scheme="een_total", physics=physics,
        barotropic_forcing_centred=True, barotropic_een_seed="nemo_kmm",
    )
    model = LatLonCGridOceanModel(grid, z, cfg)

    s1 = model.step(state, dt=_DT)   # step 0 (Euler start)
    assert np.all(np.isfinite(np.asarray(s1.T.data)))
    assert np.all(np.isfinite(np.asarray(s1.u.data)))
    s2 = model.step(s1, dt=_DT)      # step 1 (genuine leap-frog)
    assert np.all(np.isfinite(np.asarray(s2.T.data)))
    assert np.all(np.isfinite(np.asarray(s2.u.data)))


def test_n2_nemo_before_tracers_bridged_state_not_clobbered():
    """Sibling regression: a state that ALREADY carries bridged/restart
    before-level fields (kamm_twin_90d --bridge-before) must NOT be
    overwritten by the from-rest Euler-start seed -- ``_leapfrog_step``'s
    seed only fires inside the ``state.u_before is None`` branch, which a
    bridged state never enters."""
    state, model = _leapfrog_channel()
    # Simulate a bridged/restart state: distinct before-level fields (NOT
    # equal to now), which the fix must preserve untouched through step().
    bridged = state._replace(
        u_before=state.u.replace(data=state.u.data + 0.01),
        v_before=state.v.replace(data=state.v.data + 0.01),
        T_before=state.T.replace(data=state.T.data + 1.0),
        S_before=state.S.replace(data=state.S.data),
        eta_before=state.eta.replace(data=state.eta.data),
    )
    assert bridged.u_before is not None
    s = model.step(bridged, dt=_DT)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    # bridged path takes the FULL leapfrog branch (not the Euler-start
    # branch), so it must not equal the from-rest Euler-start result.
    s_fresh = model.step(state, dt=_DT)
    assert not np.allclose(np.asarray(s.T.data), np.asarray(s_fresh.T.data))


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


# ---------------------------------------------------------------------------
# Node 19 / residual #2 — thickness-weighted tracer Robert-Asselin filter
# (NEMO tra_atf_qco_lf, key_qco z*).  Truth-tier: global content conservation.
# ---------------------------------------------------------------------------

def test_thickness_weighted_asselin_conserves_content():
    """TW filter conserves globally-integrated tracer content; concentration doesn't.

    Set the per-cell CONTENT ``C0 = e3t*T`` EQUAL at the three time levels
    (T_level = C0/e3t_level) with *varying* thickness (moving z*).  Then the
    global content is identical at before/now/after, so NEMO's thickness-weighted
    filter (a time-Laplacian of content) must leave the global content unchanged
    to machine precision, while the plain concentration filter drifts by O(gamma*dη).
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _thickness_weighted_asselin,
    )
    rng = np.random.default_rng(0)
    shape = (6, 8, 4)
    gamma = 0.1
    mask = np.ones(shape)
    # three DISTINCT positive thickness fields (varying z* geometry)
    e3n = jnp.asarray(50.0 + 10.0 * rng.random(shape))
    e3b = jnp.asarray(50.0 + 10.0 * rng.random(shape))
    e3a = jnp.asarray(50.0 + 10.0 * rng.random(shape))
    # filtered thickness consistent with a plain-RA-filtered ssh: e3t is linear
    # in ssh, so the filtered thickness == plain filter of the thicknesses.
    e3f = e3n + gamma * (e3b - 2.0 * e3n + e3a)
    # EQUAL per-cell content C0 across levels -> equal global content
    C0 = jnp.asarray(1000.0 + 500.0 * rng.random(shape))
    Tn = C0 / e3n
    Tb = C0 / e3b
    Ta = C0 / e3a
    content0 = float(jnp.sum(C0))

    # thickness-weighted (NEMO) -> global content preserved to machine precision
    T_f = _thickness_weighted_asselin(Tn, Tb, Ta, e3n, e3b, e3a, e3f,
                                      gamma, jnp.asarray(mask))
    content_tw = float(jnp.sum(e3f * T_f))
    assert abs(content_tw - content0) / content0 < 1e-13

    # plain concentration form -> finite O(gamma*dη) content drift
    T_conc = Tn + gamma * (Tb - 2.0 * Tn + Ta)
    content_conc = float(jnp.sum(e3f * T_conc))
    assert abs(content_conc - content0) / content0 > 1e-6

    # GENERAL case (unequal contents): the TW filter perturbs global content by
    # EXACTLY the discrete time-Laplacian of content, gamma*(C_b-2C_n+C_a) --
    # no spurious e3t-inconsistency source.  Use independent T fields.
    Tn2 = jnp.asarray(rng.random(shape))
    Tb2 = jnp.asarray(rng.random(shape))
    Ta2 = jnp.asarray(rng.random(shape))
    T_f2 = _thickness_weighted_asselin(Tn2, Tb2, Ta2, e3n, e3b, e3a, e3f,
                                       gamma, jnp.asarray(mask))
    lhs = float(jnp.sum(e3f * T_f2 - e3n * Tn2))                  # Δ content
    rhs = float(gamma * jnp.sum(e3b * Tb2 - 2.0 * e3n * Tn2 + e3a * Ta2))
    assert abs(lhs - rhs) / (abs(rhs) + 1e-30) < 1e-12


def test_thickness_weighted_asselin_reduces_to_plain_at_fixed_volume():
    """Fixed volume (e3t time-invariant) -> TW filter == plain concentration RA.

    Backward-compat guarantee: a non-moving coordinate makes the thickness cancel
    exactly, so the content form collapses to ``T_n + gamma*(T_b-2T_n+T_a)``.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _thickness_weighted_asselin,
    )
    rng = np.random.default_rng(1)
    shape = (5, 7, 3)
    gamma = 0.1
    e3 = jnp.asarray(80.0 + 5.0 * rng.random(shape))   # same at all levels
    mask = jnp.ones(shape)
    Tn = jnp.asarray(rng.random(shape))
    Tb = jnp.asarray(rng.random(shape))
    Ta = jnp.asarray(rng.random(shape))
    T_tw = _thickness_weighted_asselin(Tn, Tb, Ta, e3, e3, e3, e3, gamma, mask)
    T_plain = Tn + gamma * (Tb - 2.0 * Tn + Ta)
    assert float(jnp.max(jnp.abs(T_tw - T_plain))) < 1e-12


def test_thickness_weighted_asselin_masks_dry_cells():
    """Below-seafloor cells (e3_f == 0) return 0, no NaN from the divide."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _thickness_weighted_asselin,
    )
    shape = (3, 3, 2)
    gamma = 0.1
    e3 = jnp.ones(shape)
    e3f = e3.at[..., -1].set(0.0)          # bottom level dry
    mask = jnp.ones(shape).at[..., -1].set(0.0)
    T = jnp.asarray(np.random.default_rng(2).random(shape))
    out = _thickness_weighted_asselin(T, T, T, e3, e3, e3f, e3f, gamma, mask)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# #1226 item 3: barotropic_forcing_centred (NEMO ln_bt_fw=.FALSE.,
# dynspg_ts.F90:392-421 wind/emp ½(before+now) + :1623-1636 drag Kbb residual)
# ---------------------------------------------------------------------------

def _sf(tau_x=0.0, tau_y=0.0, n_lat=8, n_lon=16):
    from legoesm.ocean.state import OceanSurfaceForcing
    return OceanSurfaceForcing(
        tau_x=jnp.full((n_lat, n_lon), tau_x),
        tau_y=jnp.full((n_lat, n_lon), tau_y))


def _fw(net=0.0, n_lat=8, n_lon=16):
    from legoesm.ocean.freshwater import FreshwaterForcing
    z = jnp.zeros((n_lat, n_lon))
    return FreshwaterForcing(precip=jnp.full((n_lat, n_lon), net),
                              evap=z, runoff=z, ice_fw=z)


def test_barotropic_forcing_centred_default_false():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert LatLonCGridOceanConfig.from_flat().barotropic_forcing_centred is False


def test_barotropic_forcing_centred_requires_leapfrog():
    with pytest.raises(ValueError, match="barotropic_forcing_centred"):
        _channel("forward_euler", barotropic_forcing_centred=True)
    with pytest.raises(ValueError, match="barotropic_forcing_centred"):
        _channel("ab2", barotropic_forcing_centred=True)


def test_barotropic_forcing_centred_off_is_bit_identical():
    """The flag off (default) must not change a single bit of the leapfrog
    step even with a time-varying wind/freshwater — the new code paths are
    gated Python ``if``s on the static config bool."""
    state, model_off = _leapfrog_channel(
        barotropic_forcing_centred=False, freshwater_closure="virtual_salt_flux")
    state2, model_on_but_false = _leapfrog_channel(
        barotropic_forcing_centred=False, freshwater_closure="virtual_salt_flux")
    sf1, sf2 = _sf(tau_x=0.02), _sf(tau_x=0.05)
    fw1, fw2 = _fw(net=1.0e-5), _fw(net=-2.0e-5)
    s = state
    for sf, fw in ((sf1, fw1), (sf2, fw2)):
        s = model_off.step(s, dt=_DT, surface_forcing=sf, freshwater=fw)
    s2 = state2
    for sf, fw in ((sf1, fw1), (sf2, fw2)):
        s2 = model_on_but_false.step(s2, dt=_DT, surface_forcing=sf, freshwater=fw)
    np.testing.assert_allclose(np.asarray(s.T.data), np.asarray(s2.T.data),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(s.u.data), np.asarray(s2.u.data),
                               rtol=0, atol=0)
    # the carry fields stay inert (never read) when the flag is False
    assert s.tau_x_prev is None


def test_barotropic_forcing_centred_first_step_matches_now():
    """NEMO nit000 (sbcmod.F90:568-573, no restart): 'before' is set equal to
    'now' on the very first step, so the ½(before+now) average degenerates to
    plain NOW. The leap-frog's forward-Euler-start branch runs BEFORE
    tau_x_prev exists, so step 1 under centred=True must be BIT-IDENTICAL to
    step 1 under centred=False."""
    state, model_on = _leapfrog_channel(barotropic_forcing_centred=True)
    _, model_off = _leapfrog_channel(barotropic_forcing_centred=False)
    sf = _sf(tau_x=0.07, tau_y=-0.03)
    fw = _fw(net=3.0e-5)
    s_on = model_on.step(state, dt=_DT, surface_forcing=sf, freshwater=fw)
    s_off = model_off.step(state, dt=_DT, surface_forcing=sf, freshwater=fw)
    np.testing.assert_allclose(np.asarray(s_on.T.data), np.asarray(s_off.T.data),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(s_on.u.data), np.asarray(s_off.u.data),
                               rtol=0, atol=0)
    # step 1 seeds the carry to THIS step's now-forcing (NEMO's "before:=now")
    np.testing.assert_allclose(np.asarray(s_on.tau_x_prev), np.asarray(sf.tau_x))
    np.testing.assert_allclose(np.asarray(s_on.tau_y_prev), np.asarray(sf.tau_y))


def test_barotropic_forcing_centred_second_step_averages_wind_and_emp():
    """Analytic two-step check: wind/freshwater JUMP between step 1 and step
    2. The centred step-2 momentum tendency must equal the tendency computed
    from the MANUALLY-averaged ½(before+now) forcing under centred=False —
    i.e. the internal average is exactly ½(before+now), not some other
    blend. (dynspg_ts.F90:400-401 wind, :417-421 emp)."""
    state, model_on = _leapfrog_channel(
        barotropic_forcing_centred=True, freshwater_closure="virtual_salt_flux")
    _, model_manual = _leapfrog_channel(
        barotropic_forcing_centred=False, freshwater_closure="virtual_salt_flux")
    sf1 = _sf(tau_x=0.02, tau_y=0.01)
    sf2 = _sf(tau_x=0.10, tau_y=-0.04)          # jump
    fw1 = _fw(net=1.0e-5)
    fw2 = _fw(net=-3.0e-5)                       # jump
    sf_avg = sf2._replace(tau_x=0.5 * (sf1.tau_x + sf2.tau_x),
                          tau_y=0.5 * (sf1.tau_y + sf2.tau_y))
    fw_avg = fw2._replace(precip=0.5 * (fw1.precip + fw2.precip))

    s1_on = model_on.step(state, dt=_DT, surface_forcing=sf1, freshwater=fw1)
    s2_on = model_on.step(s1_on, dt=_DT, surface_forcing=sf2, freshwater=fw2)

    s1_manual = model_manual.step(state, dt=_DT, surface_forcing=sf1, freshwater=fw1)
    s2_manual = model_manual.step(s1_manual, dt=_DT, surface_forcing=sf_avg,
                                  freshwater=fw_avg)

    # ONLY the eta/barotropic channel is centred (F_slow_u/eta); the
    # separate virtual-salt-flux tracer deposit stays at NOW in BOTH runs
    # (model_manual's step-2 freshwater=fw_avg would ALSO recentre the
    # salt-flux channel, which the model itself does not do) -- so S is
    # deliberately excluded from this comparison; see the drag/eta-only
    # scoping note on ``freshwater_eta_prev`` in state.py.
    np.testing.assert_allclose(
        np.asarray(s2_on.u.data), np.asarray(s2_manual.u.data),
        rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(s2_on.eta.data), np.asarray(s2_manual.eta.data),
        rtol=1e-11, atol=1e-11)


def test_barotropic_forcing_centred_carry_swaps_every_step():
    """NEMO sbcmod.F90:382-386 utau_b(:,:) = utauU(:,:): after step n, the
    carry becomes step n's now-forcing (ready to be averaged with step n+1's
    forcing)."""
    state, model = _leapfrog_channel(barotropic_forcing_centred=True)
    sf1, sf2 = _sf(tau_x=0.02), _sf(tau_x=0.09)
    s1 = model.step(state, dt=_DT, surface_forcing=sf1)
    np.testing.assert_allclose(np.asarray(s1.tau_x_prev), np.asarray(sf1.tau_x))
    s2 = model.step(s1, dt=_DT, surface_forcing=sf2)
    np.testing.assert_allclose(np.asarray(s2.tau_x_prev), np.asarray(sf2.tau_x))


def _leapfrog_partial_cell_channel(n_lat=8, n_lon=16, n_levels=5,
                                   H_max=3000.0, **cfg_kw):
    """Leap-frog channel on an OceanPartialCellCoordinate (DINO's kamm cards
    always use one) — ``nemo_bottom_drag_rate_faces`` (barotropic_drag_substep)
    requires it. Mirrors ``test_zdf_dynzdf_composition._partial_cell_channel``
    but wired for ``outer_integrator="leapfrog"``."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z0c = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    H_bathy = jnp.full((n_lat, n_lon), H_max * 0.62)
    z = create_partial_cell_coordinate(z0c, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    cfg_kw.setdefault("outer_integrator", "leapfrog")
    cfg_kw.setdefault("coriolis_scheme", "explicit_ab2")
    cfg_kw.setdefault("vorticity_scheme", "een_total")
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg_kw.setdefault("A_h", 2.0e4)
    cfg_kw.setdefault("A_v", 1.0e-3)
    cfg_kw.setdefault("K_v", 1.0e-4)
    cfg_kw.setdefault("n_barotropic_substeps", 8)
    cfg_kw.setdefault("enable_runtime_checks", False)
    config = LatLonCGridOceanConfig.from_flat(**cfg_kw)
    return state, LatLonCGridOceanModel(grid, z, config)


def test_barotropic_forcing_centred_runs_no_nan_with_drag_substep():
    """Full NEMO DINO composition (zdf_drag_in_matrix + zdf_baroclinic_only +
    barotropic_drag_substep + barotropic_forcing_centred all on): the
    drag-residual BEFORE-level switch does not destabilise the run."""
    state, model = _leapfrog_partial_cell_channel(
        barotropic_forcing_centred=True,
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1, bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    s = state
    sf = _sf(tau_x=0.03)
    for _ in range(4):
        s = model.step(s, dt=_DT, surface_forcing=sf)
    assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert np.all(np.isfinite(np.asarray(s.u.data)))


def test_barotropic_forcing_centred_drag_residual_uses_before_level():
    """Directly exercise the drag-residual time-level switch: build a state
    whose u_before differs sharply from u (now), and confirm the centred
    F_slow drag term is computed from u_before (not u) by comparing
    ``_step_impl``'s output against a hand-built F_slow_u using u_before
    explicitly (nemo_bottom_drag_rate_faces + the same reduction the model
    uses) — i.e. the residual (u_bot - U_bar) is Kbb-based, matching
    dynspg_ts.F90:1634-1636 (NOT :1627's Kmm form)."""
    state, model = _leapfrog_partial_cell_channel(
        barotropic_forcing_centred=True,
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1, bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    # Advance one (Euler-start) step to populate u_before, then perturb NOW
    # u sharply away from u_before so a wrong (Kmm) time-level pick shows up.
    s1 = model.step(state, dt=_DT)
    rng = np.random.default_rng(3)
    u_pert = jnp.asarray(np.asarray(s1.u.data)
                          + 0.5 * rng.standard_normal(s1.u.data.shape))
    s1_pert = s1._replace(u=s1.u.replace(data=u_pert * s1.u_mask.data[..., None]))
    s_on = model.step(s1_pert, dt=_DT)

    # Build the uncentred (Kmm/NOW) counterpart at the SAME perturbed state
    # to isolate the effect: it must differ from the centred (Kbb) result
    # whenever u_before != u (both are physically valid but distinct time
    # levels — this proves the code path actually switches, not a no-op).
    _, model_now = _leapfrog_partial_cell_channel(
        barotropic_forcing_centred=False,
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1, bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    s_off = model_now.step(s1_pert, dt=_DT)
    assert np.max(np.abs(np.asarray(s_on.u.data) - np.asarray(s_off.u.data))) > 1e-8


# ---------------------------------------------------------------------------
# #1226 item 4: barotropic_een_seed (NEMO dyn_cor_2D_init(Kmm),
# dynspg_ts.F90:355 + :1349-1379 — the frozen in-window EEN Coriolis
# coefficients are built from the Kmm=NOW thickness, not the window seed's)
# ---------------------------------------------------------------------------

def test_barotropic_een_seed_default_and_unknown_raises():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert (LatLonCGridOceanConfig.from_flat()
            .barotropic.barotropic_een_seed == "window_start")
    # unknown value raises at the substep entry (dispatch hardening) —
    # reached via a step on the explicit_substep path.
    state, model = _leapfrog_channel(barotropic_een_seed="bogus")
    with pytest.raises(ValueError, match="barotropic_een_seed"):
        model.step(state, dt=_DT)


def _een_mlf(seed):
    return _leapfrog_channel(
        barotropic_coriolis="een", barotropic_coriolis_split="live",
        barotropic_een_seed=seed)


def test_barotropic_een_seed_options_identical_when_before_equals_now():
    """With eta_before == eta (before==now), the window-start thickness IS the
    NOW thickness, so 'window_start' and 'nemo_kmm' must be BIT-IDENTICAL —
    proves nemo_kmm changes nothing except the thickness time level."""
    state, m_ws = _een_mlf("window_start")
    _, m_kmm = _een_mlf("nemo_kmm")
    s1 = m_ws.step(state, dt=_DT)          # Euler start populates *_before
    # Force before == now exactly (both eta and velocity/tracers).
    s1_eq = s1._replace(u_before=s1.u, v_before=s1.v, T_before=s1.T,
                        S_before=s1.S, eta_before=s1.eta)
    s_ws = m_ws.step(s1_eq, dt=_DT)
    s_kmm = m_kmm.step(s1_eq, dt=_DT)
    np.testing.assert_allclose(np.asarray(s_ws.u.data),
                               np.asarray(s_kmm.u.data), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(s_ws.eta.data),
                               np.asarray(s_kmm.eta.data), rtol=0, atol=0)


def test_barotropic_een_seed_sensitivity_nbb_vs_kmm():
    """With eta_before != eta (spatially varying difference), the two seeds
    build the EEN coefficients from DIFFERENT thicknesses (Nbb vs Kmm), so
    the stepped states must differ — the deviation the option closes is
    real, and the option actually switches the thickness."""
    state, m_ws = _een_mlf("window_start")
    _, m_kmm = _een_mlf("nemo_kmm")
    s1 = m_ws.step(state, dt=_DT)
    # Spatially-varying before-eta perturbation (a uniform shift would nearly
    # cancel in the EEN f/h·h structure); mask-safe, small vs H.
    rng = np.random.default_rng(11)
    eta_b = (np.asarray(s1.eta_before.data)
             + 2.0 * rng.standard_normal(s1.eta.data.shape)
             * np.asarray(s1.land_mask.data))
    s1_pert = s1._replace(
        eta_before=s1.eta_before.replace(data=jnp.asarray(eta_b)))
    s_ws = m_ws.step(s1_pert, dt=_DT)
    s_kmm = m_kmm.step(s1_pert, dt=_DT)
    assert np.max(np.abs(np.asarray(s_ws.u.data)
                         - np.asarray(s_kmm.u.data))) > 1e-10


# ------------------------------------ #1455 barotropic drag-rate time level ---

def test_barotropic_drag_rate_receives_the_now_velocity_not_u_star():
    """The production step must hand the barotropic solver the STEP-ENTRY
    (NEMO ``Kmm``) velocity for the bottom-drag rate.

    NEMO builds ``rCdU_bot`` in ``zdf_phy`` from ``uu(:,:,:,Kmm)``
    (zdfdrg.F90:174-181) at stpmlf.F90:190, i.e. BEFORE ``dyn_adv``/``dyn_vor``/
    ``dyn_ldf``/``dyn_hpg``/``dyn_spg``, and ``dyn_drg_init``
    (dynspg_ts.F90:1616) freezes it across the substep window.  legoESM calls
    the solver with ``state_mid``, whose velocity is the POST-momentum
    ``u* = u^n + dt·RHS`` (plus the Matsuno rotation), so the now level must
    travel as ``u_now``/``v_now``.  This asserts BOTH halves: the forwarded
    array IS the step-entry velocity, and it is NOT ``state_mid``'s.

    #1455: before the fix the rate was built from ``state_mid``'s velocity.
    On the DINO card that velocity differed from the now-level one by
    1.115e-01 m/s at the maximum over wet cells -- measured, not asserted
    here, from the retention lane's committed capture of every drag-helper
    call in one card step (``results/dino_1455/maps/drag_inputs_d180.npz``,
    probe commit 888d846f3).
    """
    state, model = _leapfrog_partial_cell_channel(
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1, bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    # A wind stress and a non-rest velocity so the momentum update actually
    # moves u between the step entry and the barotropic call — otherwise
    # u* == u^n and the assertion below could not fail (control on the control).
    rng = np.random.default_rng(11)
    s0 = state._replace(u=state.u.replace(
        data=jnp.asarray(0.3 * rng.standard_normal(state.u.data.shape))
        * state.u_mask.data[..., None]))

    captured = {}
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as _m
    orig = _m.barotropic_substeps_latlon_cgrid

    def _spy(state_mid, *a, **k):
        captured["u_now"] = (None if k.get("u_now") is None
                             else np.asarray(k["u_now"]))
        captured["u_mid"] = np.asarray(state_mid.u.data)
        return orig(state_mid, *a, **k)

    _m.barotropic_substeps_latlon_cgrid = _spy
    try:
        # ``_step_impl`` (not ``step``) so the spy sees concrete arrays
        # rather than JIT tracers — same escape the substep-scale spy above
        # uses.
        model._step_impl(s0, _DT, surface_forcing=_sf(tau_x=0.05))
    finally:
        _m.barotropic_substeps_latlon_cgrid = orig

    assert captured, "the barotropic solver was never called"
    assert captured["u_now"] is not None, (
        "the production step called the barotropic solver WITHOUT u_now -- the "
        "bottom-drag rate would be built from the post-momentum u*, not NEMO's "
        "Kmm velocity (zdfdrg.F90:174-181 via stpmlf.F90:190)")
    np.testing.assert_array_equal(captured["u_now"], np.asarray(s0.u.data))
    # ...and the two time levels really are distinct here, so the equality
    # above is a time-level assertion and not a tautology.
    assert np.max(np.abs(captured["u_mid"] - captured["u_now"])) > 1e-6


def test_every_bottom_drag_rate_uses_the_step_entry_velocity():
    """NEMO builds ``rCdU_bot`` ONCE per step, in ``zdf_phy`` from
    ``uu(:,:,:,Kmm)`` (zdfdrg.F90:174-181 via zdfphy.F90:277 at stpmlf.F90:190),
    and every consumer reads that one stored array: ``dyn_drg_init`` for the
    barotropic loop (dynspg_ts.F90:1616), the ``pu_RHSi`` residual (:1642), and
    ``dyn_zdf``, which only ``USE zdfdrg`` (dynzdf.F90:22) and reads the array
    at :156-159 and :296 without ever recomputing it.

    So within ONE step there is exactly one drag velocity: the step-entry one.
    This asserts that invariant over every call routed through the SHARED
    faces helper -- the barotropic loop, the dyn_drg_init residual and the
    implicit vertical-mixing matrix -- so a fourth consumer added on that
    route cannot quietly pick the wrong level.

    SCOPE, stated because the obvious wider claim would be false: it does NOT
    cover ``_tke_bottom_dirichlet``, which calls ``nemo_effective_bottom_drag_r``
    directly and so bypasses this spy entirely.  That site is still on the
    handed state's velocity, and NEMO's zdftke.F90:285 pairs the Kmm rate with
    a Kbb speed anyway, so it is a separate two-time-level question.

    Scoped to a single ``_step_impl`` so "now" is unambiguous: under the full
    leap-frog the before-level dissipation pass legitimately treats Kbb as its
    own now level.
    """
    state, model = _leapfrog_partial_cell_channel(
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1, bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    # A sheared, non-rest velocity plus wind, so the momentum update actually
    # moves u between the step entry and each drag site -- otherwise every
    # candidate velocity coincides and the assertion could not fail.
    rng = np.random.default_rng(7)
    s0 = state._replace(u=state.u.replace(
        data=jnp.asarray(0.3 * rng.standard_normal(state.u.data.shape))
        * state.u_mask.data[..., None]))

    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as _pemod
    seen = []
    orig = _pemod.nemo_bottom_drag_rate_faces

    def _spy(u, v, h_k, z_coord, config, grid):
        seen.append((np.asarray(u), np.asarray(v)))
        return orig(u, v, h_k, z_coord, config, grid)

    _pemod.nemo_bottom_drag_rate_faces = _spy
    try:
        model._step_impl(s0, _DT, surface_forcing=_sf(tau_x=0.05))
    finally:
        _pemod.nemo_bottom_drag_rate_faces = orig

    assert len(seen) >= 2, f"expected the drag helper to fire at several sites, got {len(seen)}"
    u_now, v_now = np.asarray(s0.u.data), np.asarray(s0.v.data)
    for i, (u_i, v_i) in enumerate(seen):
        assert np.array_equal(u_i, u_now), (
            f"drag call {i} of {len(seen)} was built from a velocity that is "
            f"not the step-entry (Kmm) one: max|du| = "
            f"{np.abs(u_i - u_now).max():.4e} m/s")
        assert np.array_equal(v_i, v_now), (
            f"drag call {i} of {len(seen)}: max|dv| = "
            f"{np.abs(v_i - v_now).max():.4e} m/s")
    # ...and the step really did move the velocity, so the equalities above are
    # assertions about a time level and not a rest-state tautology.
    out = model._step_impl(s0, _DT, surface_forcing=_sf(tau_x=0.05))
    # ``_step_impl`` returns either the state or (state, extras) depending on
    # the config -- and the state is itself a NamedTuple, so an isinstance
    # tuple check cannot tell them apart.  Probe for the field instead.
    _out = out if hasattr(out, "u") else out[0]
    assert np.max(np.abs(np.asarray(_out.u.data) - u_now)) > 1e-6


def test_implicit_vmix_partial_now_velocity_raises():
    """One component of the now-level velocity without the other would build
    the drag rate's ``|U|`` from two different time levels -- a plausible
    number with no error anywhere.  Rejected, both ways round (same rule the
    barotropic solver applies)."""
    state, model = _leapfrog_partial_cell_channel(
        bottom_drag_scheme="nemo_quadratic", zdf_drag_in_matrix=True,
        zdf_baroclinic_only=True, barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    for kw in ({"u_now": state.u.data}, {"v_now": state.v.data}):
        with pytest.raises(ValueError, match="BOTH u_now and v_now or NEITHER"):
            model._apply_implicit_vertical_mixing(
                state, _DT, _sf(), **kw)


def test_leapfrog_vmix_drag_rate_uses_the_step_entry_velocity():
    """N2 from review: scoping the invariant test to ``_step_impl`` leaves the
    site the shipped kamm_mlf card actually runs -- the implicit vertical-mixing
    call inside ``_leapfrog_step`` -- untested, so a regression there would pass
    green.  This covers it directly.

    Unambiguous by construction: both ``_step_impl`` calls inside
    ``_leapfrog_step`` pass ``_apply_implicit_vmix=False``, so the drag-in-matrix
    never runs on the before-level pass; the single vertical-mixing call sees the
    true Nnn state.
    """
    import traceback as _tb
    state, model = _leapfrog_partial_cell_channel(
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1, bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
        zdf_drag_in_matrix=True, zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic_solver="explicit_substep")
    sf = _sf(tau_x=0.05)
    # One Euler-start step to populate the before level, then perturb NOW u so a
    # wrong time-level pick is visible.
    s1 = model.step(state, dt=_DT, surface_forcing=sf)
    rng = np.random.default_rng(23)
    s1 = s1._replace(u=s1.u.replace(
        data=jnp.asarray(np.asarray(s1.u.data)
                         + 0.4 * rng.standard_normal(s1.u.data.shape))
        * s1.u_mask.data[..., None]))

    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as _pemod
    seen = []
    orig = _pemod.nemo_bottom_drag_rate_faces

    def _spy(u, v, h_k, z_coord, config, grid):
        seen.append((_tb.extract_stack()[-2].name, np.asarray(u)))
        return orig(u, v, h_k, z_coord, config, grid)

    _pemod.nemo_bottom_drag_rate_faces = _spy
    try:
        model._leapfrog_step(s1, _DT, surface_forcing=sf)
    finally:
        _pemod.nemo_bottom_drag_rate_faces = orig

    vmix = [u for name, u in seen if name == "_apply_implicit_vertical_mixing"]
    assert len(vmix) == 1, (
        f"expected exactly one implicit-vmix drag call on the leap-frog path, "
        f"got {len(vmix)} (sites seen: {[n for n, _ in seen]})")
    u_now = np.asarray(s1.u.data)
    assert np.array_equal(vmix[0], u_now), (
        "the leap-frog path's implicit vertical-mixing drag rate was built from "
        f"a velocity that is not the step-entry (Kmm) one: max|du| = "
        f"{np.abs(vmix[0] - u_now).max():.4e} m/s")
    # Control: some OTHER call in the same step used a different velocity, so
    # the equality above discriminates a time level rather than being trivial.
    others = [u for name, u in seen if name != "_apply_implicit_vertical_mixing"]
    assert any(not np.array_equal(u, u_now) for u in others), (
        "no drag call in this step differed from the entry velocity -- the "
        "assertion above cannot discriminate a time level here")
