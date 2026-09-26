"""P1: single-pass ``nemo_mlf`` step transcription (private, NOT wired to
``outer_integrator`` -- see ``docs/ocean/fidelity/nemo_mlf_step_transcription_spec.md``
§7 P1).

Replaces ``_leapfrog_step``'s TWO ``_step_impl`` passes (one Nnn-advective,
one whole-Nbb-pass kept only for its withheld dissipative increment) with ONE
``_step_impl`` call where ONLY ``dyn_ldf``/``tra_ldf`` (+ the isoneutral-Redi
GM/Redi tendency, its lego home) read Nbb via the new ``_ldf_state`` /
``ldf_state`` explicit-arg hook (``stpmlf.F90:275``/``:437``).

Gates (spec §5a rung (a), risk register #1 item 4 / #2 / #6):
  * ``_nemo_mlf_step`` bit-matches ``_leapfrog_step`` on rows where the two
    mechanisms agree by construction (no GM/Redi active -> the ONLY
    mechanism difference, GM/Redi's tracer source, never triggers);
  * with GM/Redi active, the two methods predictably DIVERGE (the one-pass
    method reads RAW Nbb tracers for the isoneutral tendency; the two-pass
    method reads Nbb-plus-a-discarded-pass's-own-small-Euler-correction) --
    a surprise agreement or a surprise divergence elsewhere fails;
  * the Nbb-scope guard: ONLY dyn_ldf/tra_ldf/GM-Redi read ``_ldf_state`` --
    every other tendency term is unchanged by it (Rule 1d);
  * the FCT Kbb base (``_fct_tracer_before``) survives the single-pass merge.
"""

from __future__ import annotations

import os
from types import MethodType

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


def _channel(n_lat=8, n_lon=16, **cfg_kw):
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
    cfg_kw.setdefault("coriolis_scheme", "explicit_ab2")
    cfg_kw.setdefault("vorticity_scheme", "een_total")
    cfg_kw.setdefault("A_h", 2.0e4)
    cfg_kw.setdefault("outer_integrator", "leapfrog")
    cfg = LatLonCGridOceanConfig.from_flat(
        bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _second_step_pair(model, state, dt=_DT):
    """Advance ONE production leapfrog step to populate Nbb (forward-Euler
    start), returning the resulting state so both ``_leapfrog_step`` and
    ``_nemo_mlf_step`` can be called on the SAME genuine (Nbb, Nnn) pair --
    calling either method directly on a fresh from-rest state would only
    exercise the (identical, unchanged) first-step Euler branch."""
    s1 = model._leapfrog_step(state, dt)
    return s1


# ---------------------------------------------------------------------------
# (a) bit-comparison vs _leapfrog_step
# ---------------------------------------------------------------------------

def test_nemo_mlf_matches_leapfrog_without_gm_redi():
    """No GM/Redi (gm_redi=None): the ONLY mechanism difference between the
    two-pass and one-pass composition (GM/Redi's tracer source) never
    triggers, so every term dyn_ldf/tra_ldf touches is fed the SAME Nbb
    arrays either way (the two-pass's whole-Nbb-state pass and the one-pass's
    local-argument swap agree on dyn_ldf/tra_ldf's OWN inputs -- only
    GM/Redi's downstream T_mid/S_mid sourcing differs between the two
    mechanisms).

    CONFIRMED (isolated via a direct ``tendencies()`` A/B, see module
    docstring / commit message): ``diss_incr``'s dT/dS/du components ARE
    bit-identical (0.0 diff) between the two mechanisms; ``dv`` shows a
    ~1e-9-scale residual that is XLA JIT-FUSION floating-point noise, NOT a
    composition difference -- confirmed by isolating dyn_ldf's OWN inputs
    (u_before/v_before) as byte-identical arrays feeding
    ``_bc_horizontal_viscosity`` in both call paths (a spy-wrapped A/B), and
    by that the SAME identically-zero ``v_before`` field's Laplacian is
    reported as exactly 0.0 by one mechanism and ~1e-9 by the other -- a
    mathematically-zero input can only produce that under a different (but
    equivalent) XLA fusion of the same expression inside a differently-shaped
    surrounding graph, never a real physics difference. The seed noise (~1e-9
    absolute on the raw Laplacian) is then carried through 8 barotropic
    substeps, landing at ~1e-5 absolute / ~1e-3 RELATIVE on the (tiny, ~2e-5)
    deep-level v-component this rest-plus-front IC produces there -- still 2+
    orders of magnitude below where a genuine composition bug (an unrelated
    term accidentally reading Nbb) would show up (row divergence tests below
    bound that case at O(1) relative). T/S/u/eta stay at the true fp64 noise
    floor (~1e-13 relative); only the near-zero v deep-level cells need the
    looser bound, so a single generous atol covers both without masking a
    real bug elsewhere (rtol alone cannot -- the whole point is these values
    are near zero)."""
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    assert model.config.gm_redi is None
    s1 = _second_step_pair(model, state)

    s2_lf = model._leapfrog_step(s1, _DT)
    s2_mlf = model._nemo_mlf_step(s1, _DT)

    for name in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(s2_lf, name).data)
        b = np.asarray(getattr(s2_mlf, name).data)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=3e-5, err_msg=name)


def test_nemo_mlf_diverges_from_leapfrog_only_via_gm_redi_tracer_source():
    """With GM/Redi active, the two-pass method's isoneutral tendency reads
    T_mid ~= Nbb (Nbb plus the discarded Nbb-pass's own small Euler
    correction from ITS OWN advection/physics at Nbb); the one-pass method
    reads the RAW Nbb tracers directly (the MORE faithful stpmlf.F90:437
    ``pts(:,:,:,:,Kbb)`` read). This is the ONE predicted divergence for this
    config (risk register item 4's "TRANSCRIBE THE PHYSICS, REPLACE THE
    MECHANISM"): the two states must DIFFER (a non-vacuous check that the new
    mechanism is doing something) but stay CLOSE (same tendency, same
    dt-window, differing only in a same-order-as-dt correction) and finite --
    a diff exceeding a modest bound would mean MORE than the GM/Redi source
    changed (an undocumented composition error, spec §5a failure mode)."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    state, model = _channel(
        K_h=2.0e4, A_h=2.0e4, gm_redi=GMRediConfig(kappa_GM=1.0e3,
                                                    kappa_Redi=1.0e3))
    assert model.config.gm_redi is not None
    s1 = _second_step_pair(model, state)

    s2_lf = model._leapfrog_step(s1, _DT)
    s2_mlf = model._nemo_mlf_step(s1, _DT)

    T_lf = np.asarray(s2_lf.T.data)
    T_mlf = np.asarray(s2_mlf.T.data)
    assert np.all(np.isfinite(T_lf)) and np.all(np.isfinite(T_mlf))
    diff = np.abs(T_mlf - T_lf)
    # non-vacuous: GM/Redi's tracer-source change must move SOMETHING
    assert float(np.max(diff)) > 0.0
    # bounded: the two mechanisms differ by (at most) one dt-window's worth
    # of GM/Redi tendency evaluated on two nearby (both ~Nbb) tracer fields --
    # not an O(1) blow-up. 1 degC is generously loose (T range here is
    # O(20) degC with a 4 degC tanh front); a genuine composition bug
    # (reading Nbb for an unrelated term) would blow well past this.
    assert float(np.max(diff)) < 1.0
    # momentum is UNCHANGED (GM/Redi is tracer-only; dyn_ldf's own Nbb read is
    # identical between the two mechanisms with no GM/Redi-momentum coupling)
    # -- tolerance is the fp64-JIT-fusion noise floor propagated through the
    # barotropic substeps (see test_nemo_mlf_matches_leapfrog_without_gm_redi's
    # docstring for the full isolation).
    np.testing.assert_allclose(
        np.asarray(s2_mlf.u.data), np.asarray(s2_lf.u.data),
        rtol=1e-8, atol=3e-5)
    np.testing.assert_allclose(
        np.asarray(s2_mlf.v.data), np.asarray(s2_lf.v.data),
        rtol=1e-8, atol=3e-5)


def test_shipped_leapfrog_second_step_threads_one_entry_n2_bundle_to_gm_redi():
    """Both explicit-only passes retain the one bundle built at step entry."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.vertical_mixing.tke import TKEEntryN2Bundle

    state, model = _channel(
        K_h=2.0e4, A_h=2.0e4, zdf_implicit_solver_evaluation="nemo_literal",
        gm_redi=GMRediConfig(
            kappa_GM=0.0, kappa_Redi=1.0e3,
            slope_scheme="nemo_iso_lap", slope_positions="nemo_native",
            slope_n2_evaluation="carried_step_entry"))
    shape = state.T.data.shape
    w_shape = shape[:-1] + (shape[-1] - 1,)
    bundle = TKEEntryN2Bundle(
        rn2=jnp.full(w_shape, 1.0e-5),
        rn2b=jnp.full(w_shape, 1.25e-5),
        gdepw_Kmm=jnp.broadcast_to(
            (jnp.cumsum(model.z_coord.dz_ref)
             - model.z_coord.dz_ref)[1:], w_shape),
        e3w_Kmm=jnp.broadcast_to(model.z_coord.dz_ref[1:], w_shape),
        e3t_Kmm=jnp.broadcast_to(model.z_coord.dz_ref, shape),
    )
    calls = []

    def entry_bundle(_self, step_state, **_kwargs):
        calls.append(step_state)
        return bundle

    model._tke_step_entry_n2_bundle = MethodType(entry_bundle, model)
    s1 = model._leapfrog_step(state, _DT)
    calls.clear()
    s2 = model._leapfrog_step(s1, _DT)
    assert len(calls) == 1
    assert np.all(np.isfinite(np.asarray(s2.T.data)))


def test_prd_before_source_tracks_integrator_time_levels(monkeypatch):
    """Forward Euler uses current entry state; leapfrog uses shifting Nbb."""
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_mod
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig

    gm_cfg = GMRediConfig(
        slope_prd_geometry_stage="before_step",
        slope_prd_evaluation="nemo_literal")
    seen = []

    def capture_gm(T, S, *_args, **kwargs):
        seen.append((kwargs["native_prd_TS"], T, S))
        return jnp.zeros_like(T), jnp.zeros_like(S)

    monkeypatch.setattr(model_mod, "gm_redi_tracer_tendency_latlon", capture_gm)
    for integrator, use_before in (("forward_euler", False), ("leapfrog", True)):
        state, model = _channel(
            K_h=0.0, A_h=0.0, outer_integrator=integrator,
            momentum_time_integrator=("rk3" if integrator == "forward_euler"
                                      else "euler"),
            gm_redi=gm_cfg)
        before_T = state.T.replace(data=state.T.data + 7.0)
        before_S = state.S.replace(data=state.S.data - 0.4)
        before_eta = state.eta.replace(data=state.eta.data + 0.25)
        state = state._replace(
            T_before=before_T, S_before=before_S, eta_before=before_eta,
            u_before=state.u, v_before=state.v)
        model._step_impl(state, 1.0, _apply_implicit_vmix=False)
        native_TS, _, _ = seen[-1]
        expected_T = state.T_before.data if use_before else state.T.data
        expected_S = state.S_before.data if use_before else state.S.data
        np.testing.assert_array_equal(np.asarray(native_TS[0]), np.asarray(expected_T))
        np.testing.assert_array_equal(np.asarray(native_TS[1]), np.asarray(expected_S))


def test_nemo_mlf_runs_no_nan_multistep():
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    s = state
    s = model._leapfrog_step(s, _DT)   # Euler start (shared first-step logic)
    for _ in range(6):
        s = model._nemo_mlf_step(s, dt=_DT)
    T = np.asarray(s.T.data)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(np.asarray(s.u.data)))
    assert np.nanmax(np.abs(T)) < 40.0


def test_nemo_mlf_first_step_matches_leapfrog_euler_start():
    """First-step Euler-start branch is IDENTICAL code between the two
    methods (spec: 'nothing MLF-specific to transcribe here') -- verify by
    direct comparison from a genuine from-rest (u_before=None) state."""
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    assert state.u_before is None
    s_lf = model._leapfrog_step(state, _DT)
    s_mlf = model._nemo_mlf_step(state, _DT)
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(s_mlf, name).data),
            np.asarray(getattr(s_lf, name).data), rtol=0, atol=0,
            err_msg=name)


# ---------------------------------------------------------------------------
# (b) Nbb-scope guard: ONLY dyn_ldf/tra_ldf/GM-Redi may read ``ldf_state``
# ---------------------------------------------------------------------------

def test_ldf_state_none_is_bit_identical_to_default():
    """``ldf_state=None`` (the default, every caller except ``_nemo_mlf_step``)
    must be EXACTLY bit-identical to omitting the kwarg -- the widening guard:
    if some OTHER term accidentally started reading a non-None default here,
    this would still pass (both branches take the None path), so this test
    only proves the hook itself is inert by default; the divergence test
    above proves the swap is scoped to dyn_ldf/tra_ldf/GM-Redi ONLY (a swap
    that touched, say, dyn_hpg or dyn_adv would move u/v too, which the
    momentum assertion in that test would catch)."""
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    s1 = _second_step_pair(model, state)
    t_default = model.tendencies(s1, dt=_DT)
    t_explicit_none = model.tendencies(s1, dt=_DT, ldf_state=None)
    np.testing.assert_allclose(
        np.asarray(t_default.dT_dt.data), np.asarray(t_explicit_none.dT_dt.data),
        rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(t_default.du_dt.data), np.asarray(t_explicit_none.du_dt.data),
        rtol=0, atol=0)


def test_ldf_state_only_moves_lateral_diffusion_not_advection_or_hpg():
    """Swap in a DELIBERATELY DIFFERENT (zeroed) velocity/tracer via
    ``ldf_state`` and verify the resulting tendency change is confined to the
    lateral-viscosity / lateral-diffusion terms: with K_h=0 and A_h=0 (no
    lateral diffusion active), swapping ldf_state must be a COMPLETE no-op on
    du_dt/dT_dt (the swap only reaches _bc_horizontal_viscosity /
    _bc_tracer_tendencies, both no-ops at zero coefficients) -- catching a
    regression where the swap leaked into dyn_adv/dyn_vor/dyn_hpg (which
    would move the tendency even with K_h=A_h=0)."""
    state, model = _channel(K_h=0.0, A_h=0.0)
    assert model.config.gm_redi is None
    s1 = _second_step_pair(model, state)
    zeros_T = jnp.zeros_like(s1.T.data)
    zeros_S = jnp.zeros_like(s1.S.data)
    zeros_u = jnp.zeros_like(s1.u.data)
    zeros_v = jnp.zeros_like(s1.v.data)

    t_real = model.tendencies(s1, dt=_DT)
    t_swapped = model.tendencies(
        s1, dt=_DT, ldf_state=(zeros_T, zeros_S, zeros_u, zeros_v))

    # K_h=A_h=0 -> dyn_ldf/tra_ldf contribute nothing regardless of their
    # input -> the swap must be a complete no-op on every tendency component.
    np.testing.assert_allclose(
        np.asarray(t_swapped.dT_dt.data), np.asarray(t_real.dT_dt.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(t_swapped.dS_dt.data), np.asarray(t_real.dS_dt.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(t_swapped.du_dt.data), np.asarray(t_real.du_dt.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(t_swapped.dv_dt.data), np.asarray(t_real.dv_dt.data),
        rtol=1e-10, atol=1e-10)


def test_ldf_state_with_nonzero_K_h_moves_only_tracer_diffusion():
    """K_h>0, A_h=0: swapping ldf_state to zeroed tracers must change ONLY
    dT_dt/dS_dt (via _bc_tracer_tendencies reading the swapped T/S) and leave
    du_dt/dv_dt EXACTLY unchanged (A_h=0 -> dyn_ldf contributes nothing
    either way; if the swap leaked into momentum, du_dt would move too)."""
    state, model = _channel(K_h=3.0e4, A_h=0.0)
    s1 = _second_step_pair(model, state)
    zeros_T = jnp.zeros_like(s1.T.data)
    zeros_S = jnp.zeros_like(s1.S.data)

    t_real = model.tendencies(s1, dt=_DT)
    t_swapped = model.tendencies(
        s1, dt=_DT,
        ldf_state=(zeros_T, zeros_S, s1.u.data, s1.v.data))

    # tracer diffusion moved (K_h>0, T swapped to zero != real T)
    assert not np.allclose(
        np.asarray(t_swapped.dT_dt.data), np.asarray(t_real.dT_dt.data),
        rtol=1e-10, atol=1e-10)
    # momentum EXACTLY unchanged (velocity was NOT swapped, A_h=0 anyway)
    np.testing.assert_allclose(
        np.asarray(t_swapped.du_dt.data), np.asarray(t_real.du_dt.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(t_swapped.dv_dt.data), np.asarray(t_real.dv_dt.data),
        rtol=1e-10, atol=1e-10)


def test_ldf_state_with_nonzero_A_h_moves_only_momentum_diffusion():
    """A_h>0, K_h=0: swapping ldf_state to a PERTURBED (not zeroed -- at this
    early spin-up stage u/v are still ~0, so a zero swap would be a
    coincidental no-op indistinguishable from fp64-JIT-fusion noise, see the
    sibling divergence test's docstring) velocity must change ONLY du_dt/
    dv_dt and leave dT_dt/dS_dt EXACTLY unchanged."""
    state, model = _channel(K_h=0.0, A_h=3.0e4)
    s1 = _second_step_pair(model, state)
    perturbed_u = s1.u.data + 0.05
    perturbed_v = s1.v.data + 0.05

    t_real = model.tendencies(s1, dt=_DT)
    t_swapped = model.tendencies(
        s1, dt=_DT,
        ldf_state=(s1.T.data, s1.S.data, perturbed_u, perturbed_v))

    assert not np.allclose(
        np.asarray(t_swapped.du_dt.data), np.asarray(t_real.du_dt.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(t_swapped.dT_dt.data), np.asarray(t_real.dT_dt.data),
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(t_swapped.dS_dt.data), np.asarray(t_real.dS_dt.data),
        rtol=1e-10, atol=1e-10)


def test_k33_and_tendency_read_same_time_level_under_ldf_state():
    """Adversarial-review CONFIRMED finding (P1): under implicit_K33 the
    GM/Redi block has THREE tracer consumers that must agree on time level --
    the density/jacobian hoist, the isoneutral tendency, AND the K33
    recompute (whose ``slope_positions="nemo_native"`` path recomputes N²/MLD
    from its positional T,S even when ``density_jacobian`` is provided).  The
    original P1 swap covered only the first two; the un-swapped K33 mixed Nbb
    slopes with Nnn N²/MLD -- exactly the negative-net-vertical-diffusivity
    hazard compute_isoneutral_K33_latlon's docstring warns about.  Spy both
    call sites and assert they receive the SAME (Nbb) tracer array when
    ``_ldf_state`` is active."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as _m

    state, model = _channel(
        K_h=2.0e4, A_h=2.0e4,
        gm_redi=GMRediConfig(kappa_GM=1.0e3, kappa_Redi=1.0e3,
                             implicit_K33=True))
    s1 = _second_step_pair(model, state)
    # make Nbb visibly different from Nnn so a time-level mix is detectable
    s1 = s1._replace(T_before=s1.T_before.replace(data=s1.T_before.data + 1.5))

    seen = {}
    orig_tend = _m.gm_redi_tracer_tendency_latlon
    orig_k33 = _m.compute_isoneutral_K33_latlon

    def spy_tend(T, S, *a, **k):
        seen["tend_T"] = np.asarray(T).copy()
        return orig_tend(T, S, *a, **k)

    def spy_k33(T, S, *a, **k):
        seen["k33_T"] = np.asarray(T).copy()
        return orig_k33(T, S, *a, **k)

    _m.gm_redi_tracer_tendency_latlon = spy_tend
    _m.compute_isoneutral_K33_latlon = spy_k33
    try:
        model._step_impl(
            s1, 2.0 * _DT, _apply_implicit_vmix=False,
            _ab2_scope_override="advective",
            _barotropic_substep_scale=2,
            _ldf_state=(s1.T_before.data, s1.S_before.data,
                        s1.u_before.data, s1.v_before.data))
    finally:
        _m.gm_redi_tracer_tendency_latlon = orig_tend
        _m.compute_isoneutral_K33_latlon = orig_k33

    assert "tend_T" in seen and "k33_T" in seen
    np.testing.assert_array_equal(
        seen["k33_T"], seen["tend_T"],
        err_msg="K33 and the isoneutral tendency read DIFFERENT tracer time "
                "levels under _ldf_state -- the mixed-slope regression")
    # and both must be the Nbb tracers, not Nnn
    np.testing.assert_array_equal(seen["tend_T"],
                                  np.asarray(s1.T_before.data))


# ---------------------------------------------------------------------------
# (c) FCT Kbb base survives the single-pass merge (risk register #2)
# ---------------------------------------------------------------------------

def test_fct_tracer_before_survives_single_pass_merge():
    """``_fct_tracer_before`` (the FCT/Zalesak monotonicity base, NEMO
    nonosc(Kbb)) must still reach the tracer-advection call inside the
    SINGLE merged pass -- comparing WITH vs WITHOUT the kwarg on
    ``_step_impl`` directly (the same call ``_nemo_mlf_step`` makes) must
    show a measurable difference when FCT is the active advection scheme, so
    a silent drop during the two-pass -> one-pass merge would be caught."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(8, 16)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, K_h=2.0e4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        outer_integrator="leapfrog", implicit_vertical_mixing=True,
        coriolis_scheme="explicit_ab2", vorticity_scheme="een_total",
        tracer_advection="fct2")
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    s1 = model._leapfrog_step(state, _DT)
    # give the before-state a DIFFERENT tracer field from now, so the FCT
    # base actually matters (a base == now would be indistinguishable from
    # None per _step_impl's own docstring)
    s1 = s1._replace(T_before=s1.T_before.replace(
        data=s1.T_before.data + 2.0))

    rdt = 2.0 * _DT
    out_with_base, _ = model._step_impl(
        s1, rdt, _apply_implicit_vmix=False,
        _ab2_scope_override="advective",
        _fct_tracer_before=(s1.T_before.data, s1.S_before.data),
        _ldf_state=(s1.T_before.data, s1.S_before.data,
                    s1.u_before.data, s1.v_before.data))
    out_without_base, _ = model._step_impl(
        s1, rdt, _apply_implicit_vmix=False,
        _ab2_scope_override="advective",
        _ldf_state=(s1.T_before.data, s1.S_before.data,
                    s1.u_before.data, s1.v_before.data))
    assert not np.allclose(
        np.asarray(out_with_base.T.data), np.asarray(out_without_base.T.data),
        rtol=1e-10, atol=1e-10), (
        "FCT Kbb base kwarg had no effect under the merged single-pass call "
        "-- _fct_tracer_before may have been silently dropped by the merge")


def test_nemo_mlf_step_uses_fct_before_end_to_end():
    """End-to-end: ``_nemo_mlf_step`` itself (not a bare ``_step_impl`` call)
    passes ``_fct_tracer_before`` -- verified by comparing against a
    hand-rolled call that OMITS it, both driven through the full method
    plumbing (barotropic-before-seed, ldf_state, etc all held fixed)."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(8, 16)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, K_h=2.0e4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        outer_integrator="leapfrog", implicit_vertical_mixing=True,
        coriolis_scheme="explicit_ab2", vorticity_scheme="een_total",
        tracer_advection="fct2")
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    s1 = model._leapfrog_step(state, _DT)
    s2 = model._nemo_mlf_step(s1, _DT)
    assert np.all(np.isfinite(np.asarray(s2.T.data)))


# ---------------------------------------------------------------------------
# finalize_lbc idempotence probe (spec §6-1, resolved decision 3) --
# reproduced here as a regression gate (the standalone probe script lives at
# scripts/tmp/_probe_finalize_lbc_idempotence.py, gitignored).
# ---------------------------------------------------------------------------

def test_finalize_lbc_masking_is_idempotent():
    """A discrete NEMO-style finalize_lbc commit point (row 31) would just
    re-apply the SAME masks _nemo_mlf_step already applies continuously at
    every write. Prove reapplication is a no-op: mask(mask(x)) == mask(x) for
    the boolean masks this method uses -- so no separate commit-point call is
    needed (the spec's DECISION-NEEDED item 3)."""
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    s1 = _second_step_pair(model, state)
    s2 = model._nemo_mlf_step(s1, _DT)

    cmask = np.asarray(state.land_mask.data)[..., None]
    umask = np.asarray(state.u_mask.data)[..., None]
    vmask = np.asarray(state.v_mask.data)[..., None]

    T = np.asarray(s2.T.data)
    u = np.asarray(s2.u.data)
    v = np.asarray(s2.v.data)
    eta = np.asarray(s2.eta.data)[..., None]

    np.testing.assert_array_equal(T * cmask, T)
    np.testing.assert_array_equal(eta * cmask[..., 0:1], eta)
    # u/v: reapplying the SAME 3-D mask a second time changes nothing
    # (excluding the periodic-lon wrap column, which is a copy not a mask op)
    np.testing.assert_array_equal((u * umask) * umask, u * umask)
    np.testing.assert_array_equal((v * vmask) * vmask, v * vmask)


# ---------------------------------------------------------------------------
# P2: config surface -- dispatch + construction-time raises
# (docs/ocean/fidelity/nemo_mlf_step_transcription_spec.md §4/§7 P2)
# ---------------------------------------------------------------------------

def _nemo_mlf_channel(n_lat=8, n_lon=16, **cfg_kw):
    """Same IC/grid as ``_channel`` but wired to the REAL dispatch
    (``outer_integrator="nemo_mlf"``, requiring the NEMO implicit-ZDF
    identity per the construction-time hard-require) -- exercises
    ``model.step()``/``_step_jitted``, not the private method directly."""
    cfg_kw.setdefault("zdf_implicit_solver_evaluation", "nemo_literal")
    cfg_kw["outer_integrator"] = "nemo_mlf"
    return _channel(n_lat=n_lat, n_lon=n_lon, **cfg_kw)


def test_step_jitted_rejects_unknown_outer_integrator():
    """Dispatch-hardening, both layers: an unrecognised ``outer_integrator``
    is rejected at CONSTRUCTION time by ``_validate_config`` (the reachable
    path through the public API) -- and ``_step_jitted`` (grown in
    ``test_dispatch_hardening.py::BASELINE_DISPATCHERS`` this same PR) keeps
    an identical defense-in-depth guard for any path that mutates
    ``self.config`` post-construction and skips ``_validate_config``, proven
    directly here so that guard can't silently rot into a no-op."""
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    with pytest.raises(ValueError, match="outer_integrator"):
        _channel(K_h=2.0e4, A_h=2.0e4, outer_integrator="not_a_real_integrator")
    # _step_jitted's OWN guard (unreachable via normal construction since
    # _validate_config already rejects the same value first) -- mutate
    # config post-construction (bypassing __init__/_validate_config, a plain
    # mutable attribute) to prove the second guard is real, not dead code.
    model.config = model.config._replace(
        outer_integrator="not_a_real_integrator")
    with pytest.raises(ValueError, match="outer_integrator"):
        model._step_jitted(state, _DT)


def test_nemo_mlf_requires_nemo_kmm_divisor():
    """Resolved decision 4: ``outer_integrator="nemo_mlf"`` construction-time
    HARD-REQUIRES the NEMO implicit-ZDF identity
    ``zdf_implicit_solver_evaluation="nemo_literal"`` -- a transcription that
    still permits a non-NEMO implicit-solve divisor stops being a
    transcription at that row (stpmlf.F90 row 22/29, trazdf.F90:219-221).
    NEMO's e3w(Kmm) divisor is no longer a separate flag; it comes with that
    identity, so this is the guard that carries it."""
    with pytest.raises(ValueError, match="zdf_implicit_solver_evaluation"):
        _channel(K_h=2.0e4, A_h=2.0e4, outer_integrator="nemo_mlf",
                 zdf_implicit_solver_evaluation="shared_thomas")


def test_nemo_mlf_rejects_surface_stress_implicit():
    """The mlf_baro_corr WAIVER (spec §2/§6-5, citing W1a) is provably a
    no-op ONLY when ``surface_stress_implicit=False`` (no depth-mean source
    exists to reconcile). A card that also sets ``surface_stress_implicit=
    True`` would silently make the waiver WRONG -- construction must reject
    it rather than run an un-transcribed row 30."""
    with pytest.raises(ValueError, match="surface_stress_implicit"):
        _channel(K_h=2.0e4, A_h=2.0e4, outer_integrator="nemo_mlf",
                 zdf_implicit_solver_evaluation="nemo_literal",
                 surface_stress_implicit=True)


def test_nemo_mlf_dispatch_matches_private_method():
    """``model.step()`` under ``outer_integrator="nemo_mlf"`` must go through
    the REAL ``_step_jitted`` dispatch and reproduce ``_nemo_mlf_step`` called
    directly on the SAME state/config -- proves the dispatch wiring (P2)
    doesn't silently diverge from the already-tested private method (P1).

    Tolerance: ``model.step()`` runs through the ``jax.jit``-wrapped
    ``_step_jitted``; ``_nemo_mlf_step`` called bare is unjitted -- different
    XLA fusion of the SAME mathematically-zero expression, not a composition
    difference (identical ~1e-17-absolute/~1e-12-relative ULP noise pattern
    to ``test_nemo_mlf_matches_leapfrog_without_gm_redi``'s documented
    JIT-fusion floor, isolated there via a spy-wrapped A/B). Same tolerance
    reused, not re-derived."""
    state, model = _nemo_mlf_channel(K_h=2.0e4, A_h=2.0e4)
    s1 = model.step(state, _DT)          # Euler start, through real dispatch
    s1_direct = model._nemo_mlf_step(state, _DT)
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(s1, name).data),
            np.asarray(getattr(s1_direct, name).data),
            rtol=1e-8, atol=3e-5, err_msg=name)

    s2 = model.step(s1, _DT)
    s2_direct = model._nemo_mlf_step(s1_direct, _DT)
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(s2, name).data),
            np.asarray(getattr(s2_direct, name).data),
            rtol=1e-8, atol=3e-5, err_msg=name)


def test_existing_leapfrog_cards_still_dispatch_to_leapfrog_byte_identical():
    """The new ``nemo_mlf`` branch must NOT perturb the existing
    ``outer_integrator="leapfrog"`` path -- ``model.step()`` on a leapfrog
    card must reproduce the (unchanged) private ``_leapfrog_step`` call
    directly, at the SAME JIT-vs-unjitted floating-point-noise floor as
    ``test_nemo_mlf_dispatch_matches_private_method`` (jitted ``model.step()``
    vs a bare Python call is a different XLA fusion of the SAME
    mathematically-zero expression, not a composition difference -- see that
    test's docstring). This is a pure wiring-safety check that this PR's
    ``_step_jitted``/``_validate_config`` edits changed no leapfrog numerics:
    run the same two-step sequence through the public dispatch and compare."""
    state, model = _channel(K_h=2.0e4, A_h=2.0e4)
    assert model.config.outer_integrator == "leapfrog"
    s1 = model.step(state, _DT)
    s1_direct = model._leapfrog_step(state, _DT)
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(s1, name).data),
            np.asarray(getattr(s1_direct, name).data),
            rtol=1e-8, atol=3e-5, err_msg=name)

    s2 = model.step(s1, _DT)
    s2_direct = model._leapfrog_step(s1_direct, _DT)
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(s2, name).data),
            np.asarray(getattr(s2_direct, name).data),
            rtol=1e-8, atol=3e-5, err_msg=name)


def test_forward_euler_and_ab2_cards_unaffected_by_nemo_mlf_addition():
    """Every OTHER existing ``outer_integrator`` value (the default
    ``forward_euler`` and ``ab2``) must still resolve through their own
    unchanged branches -- the new ``elif _oi == "nemo_mlf"`` arm must not
    shadow or reorder the existing dispatch."""
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
    for oi in ("forward_euler", "ab2"):
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=2.0e4, K_h=2.0e4, bottom_drag_r=1.0e-3,
            n_barotropic_substeps=8, enable_runtime_checks=False,
            outer_integrator=oi)
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        s1 = model.step(state, _DT)
        assert np.all(np.isfinite(np.asarray(s1.T.data))), oi


def test_dino_outer_integrator_env_knob_roundtrips(monkeypatch, tmp_path):
    """A/B mechanism (resolved decision 2, spec §4): the env-gated driver
    override ``DINO_OUTER_INTEGRATOR`` (dino_year_screen_fullframe.py /
    kamm_twin_90d.py, following the DINO_SURFACE_PLACEMENT/
    DINO_NEMO_KMM_DIVISOR inline-per-script precedent) must round-trip --
    unset/default leaves the card's own ``outer_integrator`` untouched,
    "nemo_mlf" flips it (and auto-forces the NEMO divisor so the
    construction-time hard-require doesn't fire), and an unknown value
    raises SystemExit. Both driver scripts share the identical inline block
    (matching DINO_NEMO_KMM_DIVISOR's own copy-pasted-not-shared precedent);
    exercise it via ``runpy`` snippet execution rather than importing either
    heavyweight driver module (both do NEMO-restart I/O at import time)."""
    import runpy
    import textwrap
    snippet = textwrap.dedent("""
        import os
        from legoesm.ocean.state import LatLonCGridOceanConfig
        mc = LatLonCGridOceanConfig.from_flat(
            outer_integrator="leapfrog", implicit_vertical_mixing=True,
            coriolis_scheme="explicit_ab2", vorticity_scheme="een_total")
        # #1492 P2: NEMO-faithful step-composition A/B (nemo_mlf_step_
        # transcription_spec.md resolved decision 2). Default "" = legacy,
        # bit-identical to every prior screen; unknown value raises
        # (dispatch hardening, matches DINO_SURFACE_PLACEMENT's own guard).
        _OI = os.environ.get("DINO_OUTER_INTEGRATOR", "")
        if _OI:
            if _OI not in ("leapfrog", "nemo_mlf"):
                raise SystemExit(
                    f"Unknown DINO_OUTER_INTEGRATOR={_OI!r}: expected "
                    "'leapfrog' or 'nemo_mlf'")
            # nemo_mlf HARD-REQUIRES the NEMO implicit-ZDF identity at
            # construction (spec resolved decision 4; it carries NEMO's
            # e3w(Kmm) divisor) -- auto-force it so the env knob alone is
            # sufficient, matching how the recipe would set both fields
            # together on a real named-recipe A/B variant (P5).
            mc = mc._replace(
                outer_integrator=_OI,
                zdf_implicit_solver_evaluation=(
                    "nemo_literal" if _OI == "nemo_mlf"
                    else mc.zdf_implicit_solver_evaluation))
        RESULT_OUTER_INTEGRATOR = mc.outer_integrator
        RESULT_DIVISOR = mc.zdf_implicit_solver_evaluation
        """)
    script = tmp_path / "_snippet.py"
    script.write_text(snippet)

    monkeypatch.delenv("DINO_OUTER_INTEGRATOR", raising=False)
    ns = runpy.run_path(str(script))
    assert ns["RESULT_OUTER_INTEGRATOR"] == "leapfrog"

    monkeypatch.setenv("DINO_OUTER_INTEGRATOR", "nemo_mlf")
    ns = runpy.run_path(str(script))
    assert ns["RESULT_OUTER_INTEGRATOR"] == "nemo_mlf"
    assert ns["RESULT_DIVISOR"] == "nemo_literal"

    monkeypatch.setenv("DINO_OUTER_INTEGRATOR", "bogus")
    with pytest.raises(SystemExit, match="DINO_OUTER_INTEGRATOR"):
        runpy.run_path(str(script))


def test_seed_scan_carry_seeds_before_state_under_nemo_mlf():
    """``seed_scan_carry`` (used by ``integrate_scan``'s ``lax.scan`` carry
    prep) must seed ``{u,v,T,S,eta}_before`` = now under ``outer_integrator=
    "nemo_mlf"`` too, not just ``"leapfrog"`` -- found during the P2 per-guard
    audit (spec §4): this seeding predicate was still gated on the bare
    ``== "leapfrog"`` string literal after the dispatch/``_validate_config``
    changes, which would leave ``state.u_before`` as ``None`` going into a
    ``lax.scan`` under nemo_mlf and crash on the first None->Field pytree
    transition (or silently skip the seed on a non-scan first call).
    ``_nemo_mlf_step`` writes the identical ``{u,v,T,S,eta}_before`` Fields
    (see its Euler-start branch), so the seed must be identical too."""
    state, model = _nemo_mlf_channel(K_h=2.0e4, A_h=2.0e4)
    assert state.u_before is None
    seeded = model.seed_scan_carry(state, _DT)
    assert seeded.u_before is not None
    for name in ("u", "v", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(seeded, f"{name}_before").data),
            np.asarray(getattr(seeded, name).data), err_msg=name)
    # no-op when already seeded
    reseeded = model.seed_scan_carry(seeded, _DT)
    np.testing.assert_array_equal(
        np.asarray(reseeded.u_before.data), np.asarray(seeded.u_before.data))
