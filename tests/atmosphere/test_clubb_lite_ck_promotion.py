"""Physically-coherent per-column promotion of the CLUBB-lite eddy-diffusivity
coefficient ``C_K`` (``K_m = C_K·l·√(wp2)``) — the coefficient the LES
eddy-diffusivity diagnosis directly informs (docs/COMPARE_REANALYSIS.md).

Verifies the promotion is non-breaking (a uniform per-column ``C_K`` reproduces
the scalar default exactly) and reaches the physics (a per-column ``C_K``
produces per-column-varying ``K_m``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

jax.config.update("jax_enable_x64", True)


def _inputs(ncol=3, nlev=12):
    rng = np.random.default_rng(0)
    p_half = np.linspace(2.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_full = jnp.asarray(np.tile(np.linspace(15000.0, 50.0, nlev), (ncol, 1)))
    z_half = jnp.asarray(np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)))
    exner = (p_full / constants.p_ref) ** constants.kappa
    theta = 290.0 + 3e-3 * np.asarray(z_full)
    T = jnp.asarray(theta * exner)
    u = jnp.asarray(8.0 + 4.0 * rng.standard_normal((ncol, nlev)))
    v = jnp.asarray(2.0 * rng.standard_normal((ncol, nlev)))
    q_v = jnp.asarray(2e-3 + 4e-3 * rng.random((ncol, nlev)))
    rho = jnp.asarray(p_full) / (constants.R_d * T)
    return dict(
        u=u, v=v, T=T, q_v=q_v, tke=jnp.full((ncol, nlev), 0.4),
        p_full=jnp.asarray(p_full), p_half=jnp.asarray(p_half),
        z_full=z_full, z_half=z_half, T_sfc=T[:, -1] + 1.0, q_sfc=q_v[:, -1],
        rho=rho, dt=300.0,
    )


def test_uniform_per_column_ck_matches_scalar():
    """A uniform (ncol,) C_K reproduces the scalar default EXACTLY (no
    regression from the broadcast_column_param wrap)."""
    kw = _inputs()
    ncol = kw["T"].shape[0]
    cfg = CLUBBLiteConfig()
    out_scalar, _ = clubb_lite_turbulence(**kw, config=cfg)

    cfg_uniform = cfg._replace(C_K=jnp.full((ncol,), cfg.C_K))
    out_uniform, _ = clubb_lite_turbulence(**kw, config=cfg_uniform)

    # A uniform per-column C_K is a strict no-op vs the scalar path (each row is
    # scaled by the identical value) ⇒ BYTE-identical, not just close.
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Km), np.asarray(out_scalar.Km))
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Kh), np.asarray(out_scalar.Kh))


def test_per_column_ck_changes_km_per_column():
    """A per-column C_K scales each column's K_m proportionally (the LES
    correction reaches the GCM diffusivity)."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    out_base, _ = clubb_lite_turbulence(**kw, config=cfg)

    # Double C_K in column 1 only.
    ck = jnp.array([cfg.C_K, 2.0 * cfg.C_K, cfg.C_K])
    out_pc, _ = clubb_lite_turbulence(**kw, config=cfg._replace(C_K=ck))

    Km_base = np.asarray(out_base.Km)
    Km_pc = np.asarray(out_pc.Km)
    # Column 0 and 2 unchanged; column 1 K_m doubled (Km = C_K·l·sqrt(wp2),
    # linear in C_K at fixed l, wp2 — same inputs).
    np.testing.assert_allclose(Km_pc[0], Km_base[0], rtol=1e-12)
    np.testing.assert_allclose(Km_pc[2], Km_base[2], rtol=1e-12)
    np.testing.assert_allclose(Km_pc[1], 2.0 * Km_base[1], rtol=1e-12)


def test_ck_diagnosis_inverts_the_real_clubb_lite_forward():
    """The LES C_K diagnosis must invert the ACTUAL ``clubb_lite_turbulence`` forward
    closure — not just a hand-written ``Km = C_K·l·√wp2`` replica (which is what
    ``test_les_closure_diagnosis.test_clubb_coefficient_exact_inverse_of_gcm_forward``
    uses).  Run the REAL GCM forward with a known C_K, then diagnose C_K back from the
    returned ``out.Km`` using the SAME mixing length (``mixing_length(z_full, l_mix_max)``)
    and wp2 the forward used — recovery to ~machine precision.

    Why a SECOND test against the real forward: the hand-written test and the diagnosis
    share the same formula, so a DIVERGENCE between ``clubb_lite.py``'s actual
    ``Km_full = C_K·l_mix·√wp2`` (line ~215) and the inverse (e.g. an edit to ``l_mix²`` on
    one side only) would leave BOTH passing while every LES-informed correction is
    systematically biased.  This test fails the moment the real forward and the inverse
    disagree.  (``tke=0.4 ≫ tke_min`` so the forward's ``wp2=max(tke,tke_min)`` is the raw
    ``tke``; the forward's ``Km`` uses the RAW ``l_mix``, not ``l_mix_safe``.)"""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        clubb_coefficient_from_diffusivity,
    )
    from legoesm.atmosphere.physics._shared import mixing_length

    kw = _inputs()
    l_mix = mixing_length(kw["z_full"], CLUBBLiteConfig().l_mix_max)
    wp2 = kw["tke"]                                      # == forward's wp2 (no clamp)

    # (1) scalar C_K (production, vertically constant).
    ck_true = 0.31
    out, _ = clubb_lite_turbulence(**kw, config=CLUBBLiteConfig(C_K=ck_true))
    ck_rec, valid = clubb_coefficient_from_diffusivity(
        out.Km, jnp.ones_like(out.Km, bool), l_mix, wp2)
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(ck_rec), ck_true, rtol=1e-6)

    # (2) per-column C_K (the LES-informed correction) — each column recovers its own value.
    ncol = kw["T"].shape[0]
    ck_col = jnp.array([0.22, 0.31, 0.55])[:ncol]
    out_pc, _ = clubb_lite_turbulence(**kw, config=CLUBBLiteConfig(C_K=ck_col))
    ck_rec_pc, valid_pc = clubb_coefficient_from_diffusivity(
        out_pc.Km, jnp.ones_like(out_pc.Km, bool), l_mix, wp2)
    assert bool(jnp.all(valid_pc))
    np.testing.assert_allclose(
        np.asarray(ck_rec_pc), np.broadcast_to(np.asarray(ck_col)[:, None], out_pc.Km.shape),
        rtol=1e-6)


def test_prt_diagnosis_inverts_the_real_clubb_lite_forward():
    """Pr_t analog of :func:`test_ck_diagnosis_inverts_the_real_clubb_lite_forward`: the LES
    Prandtl diagnosis ``Pr_t = K_m/K_h`` must invert the ACTUAL ``clubb_lite_turbulence``
    forward ``Kh_full = Km_full / Pr_t`` (`clubb_lite.py:219`, ``Kh=Kh_full`` returned
    unmodified) — not a hand-written ``Km=4, Kh=5`` replica (which is what
    ``test_les_closure_diagnosis``'s Prandtl test uses).  Run the real GCM forward with a
    known Pr_t (scalar AND per-column), diagnose Pr_t back from ``out.Km``/``out.Kh``,
    recover to ~machine precision — so a DIVERGENCE between the real ``Kh`` forward and the
    inverse (which would bias every Pr_t correction) fails here even though both
    formula-sharing tests stay green."""
    from legoesm.atmosphere.dynamics.les.les_closure_diagnosis import (
        prandtl_number_from_diffusivities,
    )

    kw = _inputs()
    # (1) scalar Pr_t (production, vertically constant).
    prt_true = 0.7
    out, _ = clubb_lite_turbulence(**kw, config=CLUBBLiteConfig(Pr_t=prt_true))
    prt_rec, valid = prandtl_number_from_diffusivities(
        out.Km, jnp.ones_like(out.Km, bool), out.Kh, jnp.ones_like(out.Kh, bool))
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(prt_rec), prt_true, rtol=1e-6)

    # (2) per-column Pr_t (the LES-informed correction) — each column recovers its own value.
    ncol = kw["T"].shape[0]
    prt_col = jnp.array([0.5, 0.7, 1.1])[:ncol]
    out_pc, _ = clubb_lite_turbulence(**kw, config=CLUBBLiteConfig(Pr_t=prt_col))
    prt_rec_pc, valid_pc = prandtl_number_from_diffusivities(
        out_pc.Km, jnp.ones_like(out_pc.Km, bool), out_pc.Kh, jnp.ones_like(out_pc.Kh, bool))
    assert bool(jnp.all(valid_pc))
    np.testing.assert_allclose(
        np.asarray(prt_rec_pc), np.broadcast_to(np.asarray(prt_col)[:, None], out_pc.Km.shape),
        rtol=1e-6)


def test_ceps_dissipation_form_and_production_scaling_match_real_clubb_lite_forward():
    """Real-forward lock for the C_eps diagnosis PREMISE, without duplicating clubb_lite's
    inline S²/N² numerics (which would violate the no-duplicate-numerics rule).

    The ``c_eps_from_budget`` diagnosis inverts clubb_lite's wp2 DISSIPATION rate
    ``diss = C_eps·√wp2/ℓ`` (giving the equilibrium ``C_eps = P·ℓ/wp2^{3/2}``).  With a
    UNIFORM wp2 the wp2 vertical diffusion is zero (no gradient), so the semi-implicit
    update is the LOCAL balance ``wp2_new = (wp2_0 + dt·P)/(1 + dt·diss)`` (same ``wp2_0``
    and the SAME production ``P`` for any C_eps — P depends on the fixed u/v/T, not C_eps).
    Running C_eps=0 gives ``wp2_p = wp2_0 + dt·P``; C_eps=c gives ``wp2_c = wp2_p/(1+dt·diss_c)``
    ⇒ ``diss_c = (wp2_p/wp2_c − 1)/dt`` is EXTRACTED from two REAL forward runs (S²/N²
    cancel) and MUST equal ``c·√wp2_0/ℓ`` (ℓ = the SAME ``mixing_length`` the forward + the
    diagnosis use).  This catches a future edit to clubb_lite's dissipation form that the
    analytic-equilibrium diagnosis test (which shares the formula) cannot.  Also locks
    production ∝ Km ∝ C_K (doubling C_K doubles the extracted P).  Neutral (constant-θ)
    sheared column ⇒ N²=0 ⇒ ``P = Km·S² > 0`` (no wp2-floor clamp)."""
    from legoesm.atmosphere.physics._shared import mixing_length

    ncol, nlev = 2, 10
    cfg = CLUBBLiteConfig()
    z_half = jnp.asarray(np.tile(np.linspace(3000.0, 0.0, nlev + 1), (ncol, 1)))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    p_half = jnp.asarray(np.tile(np.linspace(7.0e4, 1.0e5, nlev + 1), (ncol, 1)))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    exner = (p_full / constants.p_ref) ** constants.kappa
    T = jnp.asarray(290.0 * exner)                       # constant θ ⇒ N²=0  # noqa: N806
    u = jnp.asarray(np.tile(np.linspace(20.0, 0.0, nlev), (ncol, 1)))   # linear shear
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.full((ncol, nlev), 1e-3)
    rho = jnp.asarray(p_full) / (constants.R_d * T)
    wp2_0 = 0.5
    kw = dict(u=u, v=v, T=T, q_v=q_v, tke=jnp.full((ncol, nlev), wp2_0),
              p_full=jnp.asarray(p_full), p_half=p_half, z_full=z_full, z_half=z_half,
              T_sfc=T[:, -1], q_sfc=q_v[:, -1], rho=rho, dt=200.0)
    dt = kw["dt"]

    _, wp2_p = clubb_lite_turbulence(**kw, config=cfg._replace(C_eps=0.0))   # pure production
    c = 0.4
    _, wp2_c = clubb_lite_turbulence(**kw, config=cfg._replace(C_eps=c))     # + dissipation
    assert bool(jnp.all(wp2_p > cfg.tke_min)) and bool(jnp.all(wp2_c > cfg.tke_min))

    diss_extracted = (wp2_p / wp2_c - 1.0) / dt
    l_mix_safe = jnp.clip(mixing_length(z_full, cfg.l_mix_max), 1.0, None)
    diss_form = c * jnp.sqrt(wp2_0) / l_mix_safe                              # C_eps·√wp2/ℓ
    np.testing.assert_allclose(np.asarray(diss_extracted), np.asarray(diss_form), rtol=1e-9)

    # Production ∝ Km ∝ C_K (uniform wp2 ⇒ no transport ⇒ P = (wp2_p − wp2_0)/dt).
    P1 = (wp2_p - wp2_0) / dt                                                 # noqa: N806
    _, wp2_p2 = clubb_lite_turbulence(**kw, config=cfg._replace(C_eps=0.0, C_K=2.0 * cfg.C_K))
    P2 = (wp2_p2 - wp2_0) / dt                                                # noqa: N806
    np.testing.assert_allclose(np.asarray(P2), 2.0 * np.asarray(P1), rtol=1e-9)


def test_promotion_registered_and_applies():
    """C_K is registered promotable and apply_feedback_to_scheme splices it."""
    from legoesm.training.promotable_params import (
        PROMOTABLE_FIELDS,
        apply_feedback_to_scheme,
    )

    assert "clubb_lite_C_K" in PROMOTABLE_FIELDS
    cfg = CLUBBLiteConfig()
    new = apply_feedback_to_scheme(cfg, "clubb_lite_C_K", jnp.array([0.3, 0.5, 0.7]))
    assert new.C_K.shape == (3,)
    np.testing.assert_allclose(np.asarray(new.C_K), [0.3, 0.5, 0.7])


def test_uniform_per_column_prt_matches_scalar():
    """A uniform (ncol,) Pr_t reproduces the scalar default EXACTLY (the
    broadcast_column_param wrap on Pr_t is a no-op for the scalar path)."""
    kw = _inputs()
    ncol = kw["T"].shape[0]
    cfg = CLUBBLiteConfig()
    out_scalar, _ = clubb_lite_turbulence(**kw, config=cfg)

    cfg_uniform = cfg._replace(Pr_t=jnp.full((ncol,), cfg.Pr_t))
    out_uniform, _ = clubb_lite_turbulence(**kw, config=cfg_uniform)
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Kh), np.asarray(out_scalar.Kh))
    # Km is independent of Pr_t (Kh = Km/Pr_t) ⇒ unchanged either way.
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Km), np.asarray(out_scalar.Km))


def test_per_column_prt_changes_kh_per_column():
    """A per-column Pr_t scales each column's K_h inversely (Kh = Km/Pr_t); Km
    is unaffected — the LES Prandtl-number correction reaches the GCM."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    out_base, _ = clubb_lite_turbulence(**kw, config=cfg)

    prt = jnp.array([cfg.Pr_t, 2.0 * cfg.Pr_t, cfg.Pr_t])   # double Pr_t in col 1
    out_pc, _ = clubb_lite_turbulence(**kw, config=cfg._replace(Pr_t=prt))

    Kh_base, Kh_pc = np.asarray(out_base.Kh), np.asarray(out_pc.Kh)
    np.testing.assert_allclose(Kh_pc[0], Kh_base[0], rtol=1e-12)
    np.testing.assert_allclose(Kh_pc[2], Kh_base[2], rtol=1e-12)
    np.testing.assert_allclose(Kh_pc[1], 0.5 * Kh_base[1], rtol=1e-12)  # Kh halved
    # Km is unchanged by Pr_t.
    np.testing.assert_array_equal(np.asarray(out_pc.Km), np.asarray(out_base.Km))


def test_prt_promotion_registered_and_applies():
    from legoesm.training.promotable_params import (
        PROMOTABLE_FIELDS,
        apply_feedback_to_scheme,
    )

    assert "clubb_lite_Pr_t" in PROMOTABLE_FIELDS
    new = apply_feedback_to_scheme(
        CLUBBLiteConfig(), "clubb_lite_Pr_t", jnp.array([0.7, 0.9, 1.1]))
    assert new.Pr_t.shape == (3,)
    np.testing.assert_allclose(np.asarray(new.Pr_t), [0.7, 0.9, 1.1])


def _mass_weight(kw):
    """The diffusion solver's EXACT mass weight ``w_k = ρ_k·dz_layer_k`` (the
    same ``clip(|Δz_half|, 1, None)`` the scheme applies) — the weight under which
    the flux-form column integral telescopes to the boundary flux."""
    z_half = np.asarray(kw["z_half"])
    dz_layer = np.clip(np.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    return np.asarray(kw["rho"]) * dz_layer            # (ncol, nlev)


def _exner_pref(kw):
    """``θ = T·(p_ref/p)^κ`` ⇒ dθ/dt = exner_pref·dT/dt; the energy-conserving
    weight for the θ-space heat diffusion (raw Σw·dT/dt is NOT conserved).
    Matches ``implicit_vertical_diffusion_theta``'s ``clip(p_full, 1, None)``."""
    p_safe = np.clip(np.asarray(kw["p_full"]), 1.0, None)
    return (constants.p_ref / p_safe) ** constants.kappa


def _col_int(tendency, weight):
    """Per-column mass-weighted vertical integral Σ_k weight_k·tendency_k."""
    return np.sum(np.asarray(weight) * np.asarray(tendency), axis=1)   # (ncol,)


def test_per_column_ck_prt_conserves_column_integrals():
    """CONSERVATION GATE (§9 / CLAUDE.md): the LES-informed per-column parameter
    deploy must NOT leak column mass/moisture/momentum/energy.

    The interior eddy diffusion is flux-form, so the mass-weighted column-integrated
    tendency = the surface flux — and the surface fluxes are bulk-formula
    (``compute_surface_fluxes``), INDEPENDENT of C_K/Pr_t/C_eps.  So two runs
    differing ONLY in a strongly-varying per-column C_K AND Pr_t produce IDENTICAL
    column integrals of q_v, u, v (raw) and θ (energy = exner-weighted T), even
    though every per-LEVEL profile differs.  (wp2_new is C_eps-dependent BY DESIGN
    and excluded.)"""
    kw = _inputs()
    # Ensure genuinely non-zero surface fluxes so rtol is a real test (not all ~0).
    kw["T_sfc"] = kw["T"][:, -1] + 2.0
    kw["q_sfc"] = kw["q_v"][:, -1] + 3.0e-3
    cfg = CLUBBLiteConfig()
    out0, _ = clubb_lite_turbulence(**kw, config=cfg)

    # The deployed correction: a strongly-varying per-column C_K AND Pr_t.
    ck = jnp.array([0.5 * cfg.C_K, 2.0 * cfg.C_K, 1.3 * cfg.C_K])
    prt = jnp.array([0.6 * cfg.Pr_t, 1.5 * cfg.Pr_t, 0.9 * cfg.Pr_t])
    out1, _ = clubb_lite_turbulence(**kw, config=cfg._replace(C_K=ck, Pr_t=prt))

    w = _mass_weight(kw)
    we = w * _exner_pref(kw)                            # energy (θ) weight
    for name, t0, t1, weight in [
        ("moisture", out0.dq_v_dt, out1.dq_v_dt, w),
        ("u-momentum", out0.du_dt, out1.du_dt, w),
        ("v-momentum", out0.dv_dt, out1.dv_dt, w),
        ("energy (θ)", out0.dT_dt, out1.dT_dt, we),
    ]:
        np.testing.assert_allclose(
            _col_int(t1, weight), _col_int(t0, weight), rtol=1e-9, atol=1e-12,
            err_msg=f"per-column C_K/Pr_t changed the column-integrated {name} "
                    f"tendency — the deploy leaks {name}.")

    # NON-VACUOUS: BOTH coefficients genuinely reached the body. du_dt depends on
    # Km = C_K·l·√wp2 (NO Pr_t) → it isolates C_K's effect; dq_v_dt/dT_dt are
    # Kh-driven → Pr_t's effect. Checking momentum separately rules out a silently
    # dropped C_K that Pr_t's Kh-modulation would otherwise mask (Codex e2).
    assert float(np.max(np.abs(np.asarray(out1.du_dt) - np.asarray(out0.du_dt)))) > 1e-10
    assert float(np.max(np.abs(np.asarray(out1.dq_v_dt) - np.asarray(out0.dq_v_dt)))) > 1e-10
    assert float(np.max(np.abs(np.asarray(out1.dT_dt) - np.asarray(out0.dT_dt)))) > 1e-10


def test_conservation_gate_detects_a_leak():
    """SYNTHETIC-VIOLATION self-test (CLAUDE.md 'provably non-vacuous' tripwire):
    a non-flux-form column source makes the mass-weighted integral differ, so the
    equality assertion the gate relies on FAILS — proving the gate is not vacuous."""
    kw = _inputs()
    out0, _ = clubb_lite_turbulence(**kw, config=CLUBBLiteConfig())
    w = _mass_weight(kw)
    # Inject a fake non-conservative source at the top level of one column only.
    leaked = np.asarray(out0.dq_v_dt).copy()
    leaked[1, 0] += 1.0e-6                              # a spurious moisture source
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            _col_int(leaked, w), _col_int(out0.dq_v_dt, w), rtol=1e-9, atol=1e-12)


def test_uniform_per_column_ceps_matches_scalar():
    """A uniform (ncol,) C_eps reproduces the scalar default EXACTLY."""
    kw = _inputs()
    ncol = kw["T"].shape[0]
    cfg = CLUBBLiteConfig()
    _, wp2_scalar = clubb_lite_turbulence(**kw, config=cfg)
    _, wp2_uniform = clubb_lite_turbulence(
        **kw, config=cfg._replace(C_eps=jnp.full((ncol,), cfg.C_eps)))
    np.testing.assert_array_equal(np.asarray(wp2_uniform), np.asarray(wp2_scalar))


def test_per_column_ceps_leaves_current_tendencies_invariant():
    """SOUNDNESS GUARD for the conservation gate's C_eps EXCLUSION: C_eps controls
    the PROGNOSTIC wp2 dissipation (next-step wp2), NOT the current-step eddy-diffusion
    tendencies, so test_per_column_ck_prt_conserves_column_integrals correctly omits
    it.  A strongly-varying per-column C_eps changes wp2_new but leaves dq_v_dt /
    du_dt / dv_dt / dT_dt EXACTLY unchanged.  If a future change coupled C_eps to the
    current tendencies (e.g. using the C_eps-modified wp2 for K WITHIN the step), the
    gate's C_eps exclusion would silently become unsound and a per-column C_eps deploy
    could leak — this pins the decoupling so that regression fails LOUDLY."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    out0, wp2_0 = clubb_lite_turbulence(**kw, config=cfg)
    ncol = kw["T"].shape[0]
    # strongly-varying per-column C_eps (robust to any ncol via linspace, Codex).
    ceps = jnp.linspace(0.5, 2.0, ncol) * cfg.C_eps
    out1, wp2_1 = clubb_lite_turbulence(**kw, config=cfg._replace(C_eps=ceps))
    for name, t0, t1 in [("dq_v_dt", out0.dq_v_dt, out1.dq_v_dt),
                         ("du_dt", out0.du_dt, out1.du_dt),
                         ("dv_dt", out0.dv_dt, out1.dv_dt),
                         ("dT_dt", out0.dT_dt, out1.dT_dt)]:
        np.testing.assert_array_equal(
            np.asarray(t1), np.asarray(t0),
            err_msg=f"a per-column C_eps changed the current-step {name} — the "
                    f"conservation gate's C_eps exclusion is no longer sound.")
    # NON-VACUOUS, PER-COLUMN: each column's C_eps reached its OWN wp2_new (so the
    # bit-identical-tendencies result is meaningful, not C_eps being ignored).
    per_col_dwp2 = np.max(np.abs(np.asarray(wp2_1) - np.asarray(wp2_0)), axis=1)
    assert bool(np.all(per_col_dwp2 > 1e-10))     # every column changed
    assert float(np.std(per_col_dwp2)) > 1e-12    # by DIFFERENT amounts ⇒ per-column


def test_per_column_ceps_changes_wp2_per_column():
    """A per-column C_eps changes the wp2 dissipation per column (reaches the body)."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    _, wp2_base = clubb_lite_turbulence(**kw, config=cfg)
    ceps = jnp.array([cfg.C_eps, 2.0 * cfg.C_eps, cfg.C_eps])  # double in col 1
    _, wp2_pc = clubb_lite_turbulence(**kw, config=cfg._replace(C_eps=ceps))
    np.testing.assert_array_equal(np.asarray(wp2_pc)[0], np.asarray(wp2_base)[0])
    assert not np.allclose(np.asarray(wp2_pc)[1], np.asarray(wp2_base)[1])  # more diss
