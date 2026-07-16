"""Oracle-faithfulness pins for the Simplified Betts-Miller (Frierson 2007) scheme.

Oracle: Frierson, D. M. W. (2007), "The Dynamics of Idealized Convection Schemes
and Their Effect on the Zonally Averaged Tropical Circulation", J. Atmos. Sci. 64,
1959-1976 (Betts-Miller 1986 lineage).  ``sbm_convection`` is pinned to round-off
(rel 1e-12) against an INDEPENDENT numpy reimplementation of the SBM ASSEMBLY:

    T_moist  = moist adiabat from T[:, -1]           (shared, GIVEN)
    mask     = (T_moist >= T)                          (straight-through: fwd = hard)
    Newton   : T_ref <- T_trial - residual/max(jac,1), TWICE, with
               residual = Σ_lev mask·(c_pd·(T_trial−T) + L_v·(RH_ref·q_sat(T_trial)−q_v))·dp
               jac      = Σ_lev mask·(c_pd + L_v·RH_ref·dq_sat/dT)·dp
    q_ref    = RH_ref·q_sat(T_ref)
    shallow  : dq_def = Σ mask·dp·(q_ref−q_v)/Σ mask·dp; shift = max(dq_def,0);
               q_ref −= shift; T_ref += (L_v/c_pd)·shift        (Frierson shallow)
    trigger  = sigmoid(sharpness·(CAPE − threshold))   (shared, GIVEN)
    dT_dt    = trigger·mask·(T_ref−T)/tau_c ;  dq_v_dt = trigger·mask·(q_ref−q_v)/tau_c
    cond     : dq_c = max(−dq_v_dt,0)·safe_divide(col_net_drying, col_local_cond)
    gate     : ×= (col_net_drying > 0)  (Betts-Miller P>=0: convection only dries)

SCOPE: the moist adiabat, CAPE, ``cape_trigger``, ``saturation_mixing_ratio`` and
``saturation_mixing_ratio_dT`` are SHARED, separately-tested thermodynamics — the oracle
REUSES them (a legitimate given, like constants).  It INDEPENDENTLY reimplements the
SBM-SPECIFIC glue (the Newton residual/jacobian, the relaxation, the condensation rescale
INCLUDING the ``safe_divide`` threshold/mask logic — reproduced as ``ok``/``safe_den``/
``scale`` rather than reused, which is preferable: a non-circular pin of the rescale — and
the drying gate), so the pins canary that OBSERVABLE assembly, not the thermodynamics.
(The oracle also reimplements the
Frierson shallow branch — needed for the no-shallow path of the gate test — but on a
STRONGLY net-moistening sounding the downstream #771 gate zeros the output regardless of
it, so it is NOT independently pinned there; this is NOT a global forward-deadness claim
— the strict gate can open on the post-shallow round-off residual in mixed columns; see
``test_drying_gate_zeros_moistening_output``.)

TRUTH-TIERS (analytic, independent of the oracle):
  - Total water: Σ(dq_v_dt + dq_c_conv_dt)·dp/g = 0 EXACTLY in a net-drying column
    whose column condensation candidate col_local_cond > 1e-20 (the rescale is designed
    so ∫dq_c = column net drying).  Since col_net_drying ≤ col_local_cond always, this
    holds for every physically active drying column; ONLY in the degenerate corner
    col_local_cond ≤ 1e-20 does the safe_divide AD-guard floor leave a ≤1e-20 residual.
  - Enthalpy/MSE: Σ(c_pd·dT_dt + L_v·dq_v_dt)·dp = (trigger/tau_c)·[Newton residual],
    the residual driven small by the 2-iteration Newton closure (measured, not 1e-12).

Complements the behavioral ``tests/unit/test_physics_convection.py`` (which
exercises SBM via the public API for ranges/signs) with the round-off assembly pin
+ structure canaries (Newton iteration count/sign, drying gate, condensation rescale)
+ the straight-through mask regression canary (soft backward live, forward hard).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_dT
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat, compute_cape
from legoesm.atmosphere.physics.convection._triggers import cape_trigger
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection

jax.config.update("jax_enable_x64", True)

_CPD = constants.c_pd
_LV = constants.L_v
_G = constants.g


def _np(x):
    return np.asarray(x, dtype=np.float64)


def _sbm_oracle(T, qv, p_full, p_half, cfg, *, newton_iters=2, shallow=True,
                gate=True, rescale=True, newton_sign=-1.0):
    """Independent SBM assembly. Shared thermo (moist adiabat / q_sat / CAPE /
    trigger) is called as a GIVEN; the SBM glue is reimplemented here. Flags
    override the structure for the canaries."""
    T = _np(T); qv = _np(qv); p_full = _np(p_full); p_half = _np(p_half)
    dp = p_half[:, 1:] - p_half[:, :-1]
    rh = cfg.rh_ref
    T_base = T[:, -1]
    T_moist = _np(compute_moist_adiabat(jnp.asarray(T_base), jnp.asarray(p_full)))
    cape = _np(compute_cape(jnp.asarray(T), jnp.asarray(T_moist),
                            jnp.asarray(p_full), jnp.asarray(p_half)))
    mask = (T_moist >= T).astype(np.float64)

    def qsat(Tt):
        return _np(saturation_mixing_ratio(jnp.asarray(Tt), jnp.asarray(p_full)))

    def dqsat(Tt):
        return _np(saturation_mixing_ratio_dT(jnp.asarray(Tt), jnp.asarray(p_full)))

    def newton_step(T_trial):
        q_trial = rh * qsat(T_trial)
        residual = np.sum(mask * (_CPD * (T_trial - T) + _LV * (q_trial - qv)) * dp, axis=1)
        jac = np.sum(mask * (_CPD + _LV * rh * dqsat(T_trial)) * dp, axis=1)
        dT = newton_sign * residual / np.clip(jac, 1.0, None)
        return T_trial + dT[:, None]

    T_ref = T_moist
    for _ in range(newton_iters):
        T_ref = newton_step(T_ref)
    q_ref = rh * qsat(T_ref)

    if shallow:
        w_cloud = mask * dp
        W_cloud = np.clip(np.sum(w_cloud, axis=1, keepdims=True), 1e-30, None)
        dq_def = np.sum(w_cloud * (q_ref - qv), axis=1, keepdims=True) / W_cloud
        shift = np.maximum(dq_def, 0.0)
        q_ref = q_ref - shift
        T_ref = T_ref + (_LV / _CPD) * shift

    trig = _np(cape_trigger(jnp.asarray(cape), cfg.cape_threshold,
                            cfg.smooth_trigger_sharpness))
    dT_dt = trig[:, None] * mask * (T_ref - T) / cfg.tau_c
    dq_v_dt = trig[:, None] * mask * (q_ref - qv) / cfg.tau_c

    local_cond = np.maximum(-dq_v_dt, 0.0)
    col_local_cond = np.sum(local_cond * dp / _G, axis=1, keepdims=True)
    col_net_drying = np.clip(-np.sum(dq_v_dt * dp / _G, axis=1, keepdims=True), 0.0, None)
    if rescale:
        ok = np.abs(col_local_cond) > 1e-20                  # safe_divide mask (eps=1e-20)
        safe_den = np.where(ok, col_local_cond, 1.0)
        scale = np.where(ok, col_net_drying / safe_den, 0.0)
        dq_c = local_cond * scale
    else:
        dq_c = local_cond
    if gate:
        g_gate = (col_net_drying > 0.0).astype(np.float64)
        dT_dt = dT_dt * g_gate
        dq_v_dt = dq_v_dt * g_gate
        dq_c = dq_c * g_gate
    return dict(dT_dt=dT_dt, dq_v_dt=dq_v_dt, dq_c=dq_c, cape=cape, trig=trig)


def _call(T, qv, p_full, p_half, dt, cfg):
    out = sbm_convection(jnp.asarray(T), jnp.asarray(qv), jnp.asarray(p_full),
                         jnp.asarray(p_half), dt, cfg)
    return out


# ---- soundings: a conditionally-unstable (triggering, net-drying) column ----
_CFG = SBMConfig()
_P_HALF = np.array([[50e2, 100e2, 175e2, 275e2, 400e2, 525e2, 650e2, 750e2,
                     830e2, 900e2, 955e2, 1000e2]])              # (1, 12)
_P_FULL = 0.5 * (_P_HALF[:, 1:] + _P_HALF[:, :-1])              # (1, 11)
# warm, humid boundary layer under a cool troposphere -> conditional instability
_T = np.array([[218., 220., 228., 240., 252., 262., 270., 277., 283., 288., 296.]])
_QV = np.array([[2e-6, 5e-6, 3e-5, 3e-4, 1.5e-3, 3.5e-3, 6e-3, 9e-3, 1.2e-2, 1.5e-2, 1.8e-2]])
_DT = 1200.0


def test_full_output_matches_oracle():
    out = _call(_T, _QV, _P_FULL, _P_HALF, _DT, _CFG)
    ref = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG)
    assert float(out.cape[0]) > _CFG.cape_threshold          # actually triggers
    np.testing.assert_allclose(_np(out.dT_dt), ref["dT_dt"], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(_np(out.dq_v_dt), ref["dq_v_dt"], rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(_np(out.dq_c_conv_dt), ref["dq_c"], rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(_np(out.cape), ref["cape"], rtol=1e-12)
    np.testing.assert_allclose(_np(out.convective_mask), ref["trig"], rtol=1e-12)


def test_total_water_conservation_exact():
    # TRUTH-TIER: in a net-drying column with column condensation candidate
    # col_local_cond > 1e-20 (the physically active regime) the condensation rescale
    # makes Σ(dq_v_dt + dq_c_conv_dt)·dp/g = 0 EXACTLY (total water conserved).  The
    # ≤1e-20 safe_divide-floor residual (degenerate near-zero-condensation corner) is
    # documented in the header truth-tier; here col_local_cond ≫ 1e-20 so it is 0.
    out = _call(_T, _QV, _P_FULL, _P_HALF, _DT, _CFG)
    dp = _P_HALF[:, 1:] - _P_HALF[:, :-1]
    col = np.sum((_np(out.dq_v_dt) + _np(out.dq_c_conv_dt)) * dp / _G, axis=1)
    drying = np.sum(np.abs(_np(out.dq_v_dt)) * dp / _G, axis=1)
    col_cond = np.sum(_np(out.dq_c_conv_dt) * dp / _G, axis=1)      # = col_net_drying
    assert drying[0] > 0.0                                   # non-vacuous
    assert col_cond[0] > 1e-12                              # exact regime (≫ 1e-20 floor)
    assert abs(col[0]) < 1e-12 * drying[0]


def test_enthalpy_residual_small_from_newton():
    # TRUTH-TIER (approximate, Newton-limited): the 2-iteration Newton closure
    # drives the column-integrated enthalpy tendency Σ(c_pd·dT_dt + L_v·dq_v_dt)·dp
    # small vs the term sizes.  It is NOT exact to 1e-12 (2 iterations of a
    # nonlinear q_sat(T) leave ~1e-3 relative residual; a 3rd step would reach
    # ~1e-7) — the pin is that the SECOND Newton step is load-bearing.
    out = _call(_T, _QV, _P_FULL, _P_HALF, _DT, _CFG)
    dp = _P_HALF[:, 1:] - _P_HALF[:, :-1]
    net = np.sum((_CPD * _np(out.dT_dt) + _LV * _np(out.dq_v_dt)) * dp, axis=1)
    scale = np.sum(np.abs(_CPD * _np(out.dT_dt)) * dp, axis=1)
    assert scale[0] > 0.0
    assert abs(net[0]) < 1e-2 * scale[0]                     # small (approx MSE closure)
    # discriminating: a 1-iteration closure leaves a ~100x larger residual, so
    # the module's 2nd Newton step measurably tightens enthalpy conservation.
    one = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG, newton_iters=1)
    net1 = np.sum((_CPD * one["dT_dt"] + _LV * one["dq_v_dt"]) * dp, axis=1)
    assert abs(net1[0]) > 10.0 * abs(net[0])


# ---- structure canaries ----

def test_newton_iteration_count_canary():
    ref = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG)                 # 2 iters
    one = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG, newton_iters=1)  # wrong
    out = _call(_T, _QV, _P_FULL, _P_HALF, _DT, _CFG)
    np.testing.assert_allclose(_np(out.dT_dt), ref["dT_dt"], rtol=1e-12, atol=1e-18)
    assert np.max(np.abs(ref["dT_dt"] - one["dT_dt"])) > 1e-9


def test_newton_sign_canary():
    ref = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG)
    wrong = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG, newton_sign=+1.0)
    assert np.max(np.abs(ref["dT_dt"] - wrong["dT_dt"])) > 1e-6


def test_condensation_rescale_canary():
    ref = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG)
    no_rescale = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG, rescale=False)
    out = _call(_T, _QV, _P_FULL, _P_HALF, _DT, _CFG)
    np.testing.assert_allclose(_np(out.dq_c_conv_dt), ref["dq_c"], rtol=1e-12, atol=1e-20)
    assert np.max(np.abs(ref["dq_c"] - no_rescale["dq_c"])) > 1e-9


# ---- net-moistening column: Betts-Miller P>=0 (shallow branch + drying gate) ----
# a warm, DRY conditionally-unstable column: the 70%-RH moist-adiabatic reference
# is more humid than the (dry) environment, so the RAW deep refs NET-MOISTEN — the
# regime the shallow branch + drying gate exist for (Betts-Miller P>=0, issue #771).
_T_MOIST_COL = np.array([[240., 244., 251., 260., 268., 275., 281., 286., 290., 294., 299.]])
_QV_MOIST_COL = _QV * 0.03


def test_drying_gate_zeros_moistening_output():
    # The drying gate (Betts-Miller P>=0, issue #771) zeros the LOCAL tendencies
    # of a net-moistening column.  Removing the gate leaves the shallow-corrected
    # relaxation with NONZERO per-level tendencies (whose column integral the
    # shallow branch has already neutralised to ~0) — so the P>=0 *integral*
    # alone does NOT catch a missing gate; the gate's job is to zero the residual
    # LOCAL output.  Pin: module output is identically 0, oracle-no-gate is not.
    T, QV = _T_MOIST_COL, _QV_MOIST_COL
    out = _call(T, QV, _P_FULL, _P_HALF, _DT, _CFG)
    dp = _P_HALF[:, 1:] - _P_HALF[:, :-1]
    assert np.max(np.abs(_np(out.dT_dt))) == 0.0             # gate zeros the OUTPUT
    assert np.max(np.abs(_np(out.dq_v_dt))) == 0.0
    no_gate = _sbm_oracle(T, QV, _P_FULL, _P_HALF, _CFG, gate=False)
    assert np.max(np.abs(no_gate["dT_dt"])) > 1e-6           # gate load-bearing (local)
    assert abs(np.sum(no_gate["dq_v_dt"] * dp / _G)) < 1e-15  # ...but integral ~0 (shallow)
    # RAW refs (no shallow AND no gate) WOULD net-moisten the column integral —
    # the reason the shallow branch + gate exist (though the gate alone now
    # suffices for the OUTPUT; the shallow branch only neutralises the integral,
    # which the gate then makes irrelevant — see the asymmetric FOLLOW-UP below).
    raw = _sbm_oracle(T, QV, _P_FULL, _P_HALF, _CFG, gate=False, shallow=False)
    assert np.sum(raw["dq_v_dt"] * dp / _G) > 1e-9           # raw would create water
    # ASYMMETRIC forward-redundancy ON THIS STRONGLY-MOISTENING SOUNDING (documented,
    # NOT a bidirectional NOR a GLOBAL claim): the GATE is forward-observable (it zeros
    # the local output above); here the SHALLOW branch is not — with the gate downstream,
    # dropping shallow still yields output 0 (raw integral≫0 => col_net_drying=
    # clip(-int,0)=0 => gate off).  This does NOT prove global forward-deadness: in
    # mixed/near-boundary columns the post-shallow round-off residual can open the strict
    # gate (the behavioral test_physics_convection.py has a nonzero shallow regime).
    raw_gated = _sbm_oracle(T, QV, _P_FULL, _P_HALF, _CFG, shallow=False)  # gate ON
    assert np.max(np.abs(raw_gated["dT_dt"])) == 0.0         # no-shallow+gate == 0 (this column)


# ---- trigger off ----

def test_trigger_off_stable_column():
    # A stable column (moist adiabat below environment ~ no CAPE) -> trigger ~0.
    T_stable = np.array([[300., 296., 290., 283., 275., 266., 256., 245., 233., 222., 215.]])
    qv = np.array([[2e-2, 1.6e-2, 1.2e-2, 8e-3, 5e-3, 3e-3, 1.5e-3, 5e-4, 1e-4, 2e-5, 5e-6]])
    out = _call(T_stable, qv, _P_FULL, _P_HALF, _DT, _CFG)
    assert float(out.cape[0]) <= _CFG.cape_threshold
    assert float(np.max(np.abs(_np(out.dT_dt)))) < 1e-8       # trigger sigmoid ~0


# ---- cloud_mask straight-through + differentiability ----

def test_cloud_mask_forward_is_hard_step():
    # The straight-through estimator's FORWARD value equals the hard step
    # (T_moist >= T); only the gradient is softened.  Pin the forward mask that
    # scales the tendencies is the hard 0/1 in BOTH directions: dT_dt is EXACTLY 0
    # where T_moist<T (mask kills it) AND nonzero exactly at the cloud (hard==1)
    # levels the reference relaxes.
    out = _call(_T, _QV, _P_FULL, _P_HALF, _DT, _CFG)
    T_moist = _np(compute_moist_adiabat(jnp.asarray(_T[:, -1]), jnp.asarray(_P_FULL)))
    hard = (T_moist >= _T)
    assert np.any(~hard)                                     # non-vacuous: cloud HAS a top
    assert np.all(_np(out.dT_dt)[~hard] == 0.0)              # zero OUTSIDE cloud
    ref = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG)
    inside = np.abs(ref["dT_dt"]) > 1e-12                    # levels the ref convects
    assert np.any(inside & hard)                            # non-vacuous: cloud DOES convect
    assert np.all(hard[inside])                             # convecting levels are hard==1
    assert np.all(_np(out.dT_dt)[inside] != 0.0)            # module nonzero INSIDE cloud


def test_mask_soft_backward_live_and_forward_hard_canary():
    # STRONG REGRESSION CANARY (NOT a proof) for the straight-through mask, on the
    # ACTUAL module.  ``cloud_mask_sharpness`` enters sbm_convection ONLY through the
    # mask's soft sigmoid (sbm.py:162, soft = sigmoid(sharpness*(T_moist-T))) — NOT
    # hard, Newton, q_sat, CAPE, trigger, the condensation rescale, or the gate — so
    # varying it holds EVERY branch state (hard mask values, clips, gate) fixed and
    # probes only the mask.  Two faces:
    #   * FORWARD is hard: cloud_mask = soft + stop_gradient(hard - soft) = hard,
    #     INDEPENDENT of sharpness => the module OUTPUT (all THREE tendency tensors)
    #     is invariant under sharpness and equals the HARD-mask oracle at every sampled
    #     sharpness (bit-exact here).  This RULES OUT a pure sigmoid forward and the
    #     stop_gradient(a*(sharpness-s0)^2 + ...) forward a single-point FD=0 could not
    #     (codex R3) — such forwards DIVERGE from the hard oracle at some sharpness.
    #   * BACKWARD is live: jax.grad wrt sharpness != 0 (a pure-hard mask, no soft term,
    #     gives EXACTLY 0).
    # SCOPE (codex R4): this is a canary, NOT a proof.  Finite sampling cannot rule out
    # a forward term that is zero at all sampled points, and grad!=0 does not pin the
    # EXACT surrogate derivative sigma'*(T_moist-T) (e.g. sigmoid(2*sharpness*ΔT) keeps
    # the hard forward and a nonzero gradient).  The exact backward is not scalar-
    # isolable through the full assembly (the mask enters Newton/relaxation/condensation
    # /gate nonlinearly); the forward-hard VALUE is pinned exactly by the assembly match
    # in test_full_output_matches_oracle.
    ref = _sbm_oracle(_T, _QV, _P_FULL, _P_HALF, _CFG)         # HARD mask, sharpness-free
    for sharp in (0.5, 1.3, 2.0, 17.3, 100.0):                 # forward hard => output invariant
        cfg = _CFG._replace(cloud_mask_sharpness=sharp)
        out_s = _call(_T, _QV, _P_FULL, _P_HALF, _DT, cfg)
        np.testing.assert_allclose(_np(out_s.dT_dt), ref["dT_dt"], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(_np(out_s.dq_v_dt), ref["dq_v_dt"], rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(_np(out_s.dq_c_conv_dt), ref["dq_c"], rtol=1e-12, atol=1e-20)

    def loss(sharp):
        cfg = _CFG._replace(cloud_mask_sharpness=sharp)
        out = sbm_convection(jnp.asarray(_T), jnp.asarray(_QV), jnp.asarray(_P_FULL),
                             jnp.asarray(_P_HALF), _DT, cfg)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    g = float(jax.grad(loss)(jnp.asarray(float(_CFG.cloud_mask_sharpness))))
    assert math.isfinite(g)
    assert abs(g) > 1e-12                                      # soft backward LIVE (pure-hard => 0)


def test_gradient_wrt_T_is_finite():
    # AD-safety (honest scope, NOT a surrogate-value pin): the full pipeline — the
    # straight-through mask, the max/clip kinks, safe_divide, the hard gate — yields
    # a FINITE gradient wrt the physical input T (no NaN trap).  The mask's soft
    # backward VALUE is pinned separately above; this only guarantees jax.grad works.
    def loss(scale):
        out = sbm_convection(jnp.asarray(_T) * scale, jnp.asarray(_QV),
                             jnp.asarray(_P_FULL), jnp.asarray(_P_HALF), _DT, _CFG)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    assert math.isfinite(float(jax.grad(loss)(jnp.asarray(1.0))))
