"""Faithfulness pins for the entraining mass-flux convective plume.

Target: ``plume_convection`` in
``legoesm.ocean.physics.convection.plume`` — a surface-triggered downward
convective plume that entrains ambient water while descending and detrains
heat/salt into the column.

Most-trustful source
--------------------
Entraining mass-flux plume (Paluszkiewicz & Romea 1997 give the general
entraining ocean-plume context; this pins legoESM's specific closure, not that
paper's exact scheme).  The plume properties satisfy
``dT_plume/dz = -epsilon (T_plume - T_env)``, whose EXACT solution over a layer
of thickness ``dz`` (with a layerwise-constant environment) is the convex update

    T_plume <- (1 - e) T_plume + e T_env,   e = 1 - exp(-epsilon*dz)  in [0, 1),

(NOT the first-order ``epsilon*dz``, which exceeds 1 and flips sign for thick
layers).  The detrainment tendency is
``dT/dt = w_plume * alpha_plume * epsilon * (T_plume - T_env)`` at each level,
gated by a smooth active mask on the plume-minus-environment density anomaly, and
the surface-fed heat/salt is conserved per closed column by a level-0 correction.

The existing test_plume_convection.py is behavioral (0 exact-magnitude
assertions).  This pins the exact entrainment update, the detrainment tendency,
the column-conservation truth tier, the active/surface-trigger gates, and the
T_excess source offset — using a CONTROLLED ``eos_fn`` so the density-dependent
gate is exercised independently of the EOS.

Certification (test-only):
1. Exact entrainment convex update (recovered from dT/dt) vs an independent
   oracle, incl. the thick-layer exact-vs-linear canary (T_plume stays bounded).
2. Detrainment tendency = w_plume*alpha_plume*epsilon*(T_plume-T_env)*active.
3. TRUTH TIER: sum_k dT/dt[k]*dz[k] == 0 (closed-column heat/salt conservation).
4. Gates: plume active only when denser than env (controlled eos), surface
   trigger, dry-column zero; T_excess source offset; plumbing; differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.plume import plume_convection
from legoesm.ocean.vertical import OceanZStarCoordinate

jax.config.update("jax_enable_x64", True)

_CFG = PlumeConfig()          # epsilon=1e-3, alpha_plume=0.1, w_plume_min=0.01,
#                               T_excess=0.05, active_sigmoid_sharpness=1e4
_DZ = jnp.asarray([10.0, 20.0, 30.0, 2000.0])     # last layer THICK (eps*dz = 2)


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _zcoord(dz_ref):
    dz = np.asarray(dz_ref, dtype=float)
    z_half = np.concatenate([[0.0], -np.cumsum(dz)])          # (nlev+1,)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])                 # (nlev,)
    dz_half = np.diff(z_full, prepend=z_full[0])
    return OceanZStarCoordinate(
        n_levels=len(dz), H_max=float(np.sum(dz)),
        z_full_ref=_a(z_full), z_half_ref=_a(z_half),
        dz_ref=_a(dz), dz_half_ref=_a(dz_half))


def _dense_eos(T, S, p):      # plume ALWAYS denser than the (1000) env -> active
    return jnp.full(jnp.shape(T), 2000.0, dtype=T.dtype)


def _light_eos(T, S, p):     # plume ALWAYS lighter -> inactive
    return jnp.full(jnp.shape(T), 500.0, dtype=T.dtype)


def _run(T, S, cfg=_CFG, dz=_DZ, jac=1.0, eos=_dense_eos, rho_env=1000.0,
         p_hydro=None):
    T = _a(T)
    S = _a(S)
    nlev = T.shape[-1]
    rho = jnp.full((nlev,), rho_env, dtype=jnp.float64)
    if p_hydro is None:
        p_hydro = _a(np.arange(nlev) * 1.0e5)                 # increasing with depth
    out = plume_convection(T, S, rho, p_hydro, _zcoord(dz), _a(jac), cfg, eos_fn=eos)
    return np.asarray(out.dT_dt), np.asarray(out.dS_dt), np.asarray(out.convection_flag)


def _plume_oracle(T, S, cfg, dz, jac):
    """Independent entraining-plume detrainment tendency (active everywhere)."""
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    dz_actual = np.asarray(dz, dtype=float) * jac
    nlev = T.shape[-1]
    Tp = T[0] - cfg.T_excess
    Sp = S[0]
    dT = np.zeros(nlev)
    dS = np.zeros(nlev)
    k_coeff = cfg.w_plume_min * cfg.alpha_plume * cfg.epsilon
    for k in range(1, nlev):
        e = -np.expm1(-cfg.epsilon * dz_actual[k])            # 1 - exp(-eps*dz)
        Tp = (1.0 - e) * Tp + e * T[k]
        Sp = (1.0 - e) * Sp + e * S[k]
        dT[k] = k_coeff * (Tp - T[k])
        dS[k] = k_coeff * (Sp - S[k])
    # level-0 conservation correction.
    dT[0] = -np.sum(dT * dz_actual) / dz_actual[0]
    dS[0] = -np.sum(dS * dz_actual) / dz_actual[0]
    return dT, dS


# ---------------------------------------------------------------------------
# 1. Exact entrainment + detrainment tendency.
# ---------------------------------------------------------------------------
def test_detrainment_tendency_matches_oracle():
    T = [10.0, 8.0, 6.0, 4.0]
    S = [35.0, 35.2, 35.4, 35.6]
    dT, dS, _ = _run(T, S)
    oT, oS = _plume_oracle(T, S, _CFG, _DZ, 1.0)
    np.testing.assert_allclose(dT, oT, rtol=1e-11)
    np.testing.assert_allclose(dS, oS, rtol=1e-11)


def test_exact_entrainment_not_linear_thick_layer():
    # Recover T_plume at the THICK bottom layer (eps*dz = 1e-3*2000 = 2) from the
    # tendency: T_plume[k] = T[k] + dT[k]/(w*alpha*eps).  The exact update gives a
    # CONVEX combination -> T_plume stays between the incoming plume value and
    # T_env; a linear entrain = eps*dz = 2 would EXTRAPOLATE (outside that range).
    T = [10.0, 8.0, 6.0, 4.0]
    S = [35.0, 35.0, 35.0, 35.0]
    dT, _, _ = _run(T, S)
    kc = _CFG.w_plume_min * _CFG.alpha_plume * _CFG.epsilon
    # oracle detrainment (dT array) -> reconstruct T_plume above the bottom layer:
    oT_full, _ = _plume_oracle(T, S, _CFG, _DZ, 1.0)
    # reconstruct T_plume at k=3 from the module tendency:
    tp3 = T[3] + dT[3] / kc
    e3 = -np.expm1(-_CFG.epsilon * _DZ[3])
    assert 0.0 < float(e3) < 1.0                              # exact factor in [0,1)
    # convex combination -> bounded by [T_env(4.0), incoming plume]; a linear
    # e=2 would drive tp3 below 4.0 (extrapolation).
    tp2 = T[2] + oT_full[2] / kc                              # incoming plume value
    lo, hi = sorted([tp2, T[3]])
    assert lo - 1e-9 <= tp3 <= hi + 1e-9


def test_t_excess_sets_source_parcel():
    # The plume source is COLDER than the surface by T_excess: T_plume_init =
    # T[0]-T_excess.  With eps*dz[1] small the first detrainment reflects it.
    T = [10.0, 10.0, 10.0, 10.0]                              # uniform env
    dT, _, _ = _run(T, [35.0] * 4)
    # oracle with the SAME T_excess must match; a wrong-sign (warm) excess flips dT[1].
    oT, _ = _plume_oracle(T, [35.0] * 4, _CFG, _DZ, 1.0)
    np.testing.assert_allclose(dT, oT, rtol=1e-11)
    assert dT[1] < 0.0            # colder plume than uniform env -> cools level 1


# ---------------------------------------------------------------------------
# 2. Column-conservation truth tier.
# ---------------------------------------------------------------------------
def test_column_heat_salt_conservation():
    # sum_k dT/dt[k]*dz[k] == 0 (closed column; level-0 correction).
    T = [12.0, 9.0, 7.0, 5.0]
    S = [34.0, 35.0, 35.5, 36.0]
    dT, dS, _ = _run(T, S)
    dz = np.asarray(_DZ)
    np.testing.assert_allclose(np.sum(dT * dz), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.sum(dS * dz), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Gates: active / surface-trigger / dry; plumbing; differentiability.
# ---------------------------------------------------------------------------
def test_inactive_when_plume_lighter_than_env():
    # A controlled EOS making the plume LIGHTER (500 < 1000 env): the surface
    # trigger and the active mask both vanish -> zero tendency + zero flag.
    dT, dS, flag = _run([10.0, 8.0, 6.0, 4.0], [35.0] * 4, eos=_light_eos)
    assert np.all(dT == 0.0) and np.all(dS == 0.0) and np.all(flag == 0.0)


def test_surface_trigger_gates_a_dense_in_plume():
    # Independently certify the SURFACE trigger: an EOS where the (unentrained)
    # SURFACE parcel is LIGHT at level 1 -> surface_unstable False, but the
    # ENTRAINED in-plume parcel is DENSE.  If the surface trigger is respected the
    # plume never initialises (init_active=0) -> zero output, EVEN THOUGH the
    # in-plume density gate would be active.  A mutant that dropped surface_unstable
    # (always init active) would give a nonzero tendency here.
    def _eos_surface_light(T, S, p):
        # surface parcel enters at T=100 (unentrained); the descending plume is
        # entrained toward the cold environment so T_plume < 99.95 -> dense.
        return jnp.where(T >= 99.95, 500.0, 2000.0)
    dT, dS, flag = _run([100.0, 0.0, 0.0, 0.0], [35.0] * 4, eos=_eos_surface_light)
    assert np.all(dT == 0.0) and np.all(dS == 0.0) and np.all(flag == 0.0)


def test_dry_column_zero():
    # jacobian = 0 -> dz_top = 0 -> the wet mask zeros the WHOLE column incl. the
    # convection flag (no NaN).
    dT, dS, flag = _run([10.0, 8.0, 6.0, 4.0], [35.0] * 4, jac=0.0)
    assert np.all(dT == 0.0) and np.all(dS == 0.0) and np.all(flag == 0.0)
    assert np.all(np.isfinite(dT)) and np.all(np.isfinite(dS))


def test_config_plumbing_all_three_coefficients():
    # dT = w_plume_min*alpha_plume*epsilon*(T_plume-T_env), with epsilon ALSO in
    # the entrainment recurrence.  Pin each coefficient independently vs the oracle.
    T = [10.0, 8.0, 6.0, 4.0]
    dT1, _, _ = _run(T, [35.0] * 4)
    # w_plume_min: pure prefactor -> exact 2x.
    dT_w, _, _ = _run(T, [35.0] * 4, cfg=_CFG._replace(w_plume_min=2.0 * _CFG.w_plume_min))
    np.testing.assert_allclose(dT_w, 2.0 * dT1, rtol=1e-11)
    # alpha_plume: pure prefactor -> exact 3x.
    dT_a, _, _ = _run(T, [35.0] * 4, cfg=_CFG._replace(alpha_plume=3.0 * _CFG.alpha_plume))
    np.testing.assert_allclose(dT_a, 3.0 * dT1, rtol=1e-11)
    # epsilon: enters BOTH the recurrence and the prefactor -> match the oracle
    # recomputed with the new epsilon (not a simple scaling).
    cfg_e = _CFG._replace(epsilon=5.0 * _CFG.epsilon)
    dT_e, _, _ = _run(T, [35.0] * 4, cfg=cfg_e)
    oe, _ = _plume_oracle(T, [35.0] * 4, cfg_e, _DZ, 1.0)
    np.testing.assert_allclose(dT_e, oe, rtol=1e-11)
    assert not np.allclose(dT_e, 5.0 * dT1)         # genuinely nonlinear in epsilon


def test_inplume_gate_deactivates_light_plume():
    # Complement of the surface-trigger test: an EOS where the SURFACE parcel is
    # DENSE (trigger True, init_active=1) but the entrained IN-PLUME parcel is
    # LIGHT -> the in-plume density gate must switch active off -> zero output.
    # A mutant that skipped the scan's sigmoid update would leave active=1 (nonzero).
    def _eos_plume_light(T, S, p):
        return jnp.where(T >= 99.95, 2000.0, 500.0)   # surface(T=100) dense; plume light
    dT, dS, flag = _run([100.0, 0.0, 0.0, 0.0], [35.0] * 4, eos=_eos_plume_light)
    assert np.all(dT == 0.0) and np.all(dS == 0.0) and np.all(flag == 0.0)


def test_surface_trigger_uses_level1_pressure():
    # The surface trigger displaces the surface parcel to LEVEL-1 pressure:
    # eos_fn(T[0], S[0], p_hydro[1]).  A pressure-sensitive EOS that is dense only
    # at p >= 5e4 makes the correct p[1]=1e5 comparison fire (plume active,
    # nonzero) while the erroneous level-0 (p[0]=0) comparison would NOT (zero).
    def _eos_p(T, S, p):
        return jnp.where(p >= 5.0e4, 2000.0, 500.0)
    p_hydro = _a([0.0, 1.0e5, 2.0e5, 3.0e5])
    dT, _, flag = _run([10.0, 8.0, 6.0, 4.0], [35.0] * 4, eos=_eos_p, p_hydro=p_hydro)
    assert np.any(dT != 0.0) and np.any(flag > 0.0)   # level-1 pressure -> triggered


def test_convection_flag_active_positive():
    # Positive-case flag canary: a dense, surface-unstable column has active ~ 1 at
    # every interface (a mutant returning all-zero flag would fail).
    _, _, flag = _run([10.0, 8.0, 6.0, 4.0], [35.0] * 4)
    np.testing.assert_allclose(flag, 1.0, rtol=1e-6)


def test_differentiable_through_sigmoid_gate():
    # Isolate the SMOOTH active gate by differentiating d(sum dT_dt)/dS with an
    # S-DEPENDENT eos, holding T fixed.  The heat tendency
    # dT_dt = w*alpha*eps*(T_plume - T_env)*active depends on S ONLY through
    # ``active = sigmoid(delta_rho*s)`` (T_plume/T_env are S-independent), so a
    # SMOOTH gate gives a nonzero S-gradient while a HARD density switch would give
    # exactly 0 away from the threshold.  Gentle sharpness keeps the sigmoid in its
    # transition.
    cfg = _CFG._replace(active_sigmoid_sharpness=0.5)

    def _eos_sal(T, S, p):
        return 1000.0 + (36.0 - S)               # denser (active) when fresher; delta_rho ~ O(1)

    def loss(S):
        out = plume_convection(
            _a([10.0, 8.0, 6.0, 4.0]), S, jnp.full((4,), 1000.0),
            _a(np.arange(4) * 1e5), _zcoord(_DZ), _a(1.0), cfg, eos_fn=_eos_sal)
        return jnp.sum(out.dT_dt)
    g = jax.grad(loss)(_a([35.0] * 4))
    # S enters dT_dt only via the sigmoid gate -> nonzero here iff the gate is smooth.
    assert jnp.all(jnp.isfinite(g)) and jnp.any(g != 0.0)
