"""Faithfulness pins for the Manabe-Smagorinsky-Strickler (1965) DCA path.

Target: ``_manabe_dca_convection`` / ``_adjust_one_iteration`` in
``legoesm.atmosphere.physics.convection.dca`` — the default ``variant="manabe"``
pairwise moist-adiabatic convective adjustment.

Reference (oracle)
------------------
Manabe, Smagorinsky & Strickler (1965), Mon. Wea. Rev. 93, 769-798 supplies
the *concept*: where a layer pair is more unstable than the reference adiabat,
relax it toward the reference lapse rate while conserving column enthalpy and
removing super-saturation as precipitation.  The legoESM implementation is a
specific *differentiable* discretization of that concept, whose stated algebra
(the module docstring / inline comments) is:

  * per adjacent pair (below = higher pressure, upper = lower pressure), a 2x2
    solve imposing simultaneously the moist-adiabatic target lapse and
    mass-weighted (dry) enthalpy conservation
        T_below_new - T_upper_new = gamma_m * dp_pair                  (lapse)
        dp_b*T_below_new + dp_u*T_upper_new = dp_b*T_below + dp_u*T_upper  (dry E)
  * a smooth blend ``blend = sigmoid(s*instability) * mixing_fraction *
    strat_gate`` between the old and target states;
  * super-saturation removed via ``q_new = min(q, q_sat(T_adj))``;
  * a uniform latent-heat warming closing the *moist* static energy budget
        c_p * <dT> + L_v * <dq> = 0   (column mean over the pair);
  * a bottom-to-top sweep, ``n_iterations`` times;
  * a per-column CAPE gate and a column-conservative cloud-water rescaling.

Non-circularity
---------------
The oracle here REUSES the separately-tested shared helpers
(``moist_adiabat_lapse_rate``, ``saturation_mixing_ratio``,
``stratosphere_mass_flux_gate``, ``compute_cape``, ``cape_trigger``,
``safe_divide``) as GIVENs — the SBM/CENTURY shared-helper pattern — and
independently reimplements ONLY the scheme-specific assembly (the sweep, the
2x2 solve, the blend, the moisture removal, the latent-heat closure, the
driver gating + cloud-water rescaling) in a structurally different form:

  * an explicit bottom-to-top Python loop over pairs in the ORIGINAL
    top-to-bottom level ordering, versus the module's reversed ``jax.lax.scan``;
  * the 2x2 solve written as a mass-fraction distribution about the weighted
    mean, ``T = mean +/- (dp_other/total_dp) * gamma * dp_pair`` — algebraically
    the unique solution of the two stated equations but a DIFFERENT expression
    than the module's ``(enthalpy - dp_below*gamma*dp_pair)/total_dp`` form, so a
    shared transcription error would surface as a mismatch.

The columns use a NONUNIFORM, per-column interface-pressure grid so
``dp_below != dp_upper`` for every pair — the mass weighting is genuinely
exercised (a swapped-``dp`` or equal-weight defect would break the pin).  The
oracle-INDEPENDENT truth-tiers (column moist static energy; total water) certify
the physics without the oracle at all; the perturbation canaries prove each
assembled term (latent heat, moist target, level assignment, stratosphere gate,
CAPE gate) is load-bearing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.convection.dca import (
    _adjust_one_iteration,
    _manabe_dca_convection,
    dca_convection,
)
from legoesm.atmosphere.physics.convection.config import DCAConfig
from legoesm.atmosphere.physics.thermodynamics import (
    moist_adiabat_lapse_rate,
    compute_cape,
)
from legoesm.atmosphere.physics.convection._triggers import cape_trigger
from legoesm.atmosphere.physics.convection.mass_flux import (
    stratosphere_mass_flux_gate,
)
from legoesm.atmosphere.physics._shared import safe_divide

jax.config.update("jax_enable_x64", True)  # matches test_sbm_faithful convention

_DT = 1800.0  # s


# ---------------------------------------------------------------------------
# Inputs: NONUNIFORM, per-column interface pressures (dp_below != dp_upper for
# every pair, distinct per column), saturated + conditionally unstable so the
# adjustment AND the moisture branch both fire.
# ---------------------------------------------------------------------------
def _make_columns(t_sfc, *, nlev=12, t_top=200.0, supersat=1.05, expo=None):
    """Warm, near-saturated, conditionally-unstable columns on a stretched grid.

    ``p_half`` follows ``xi**expo`` (xi linear in [0,1]) so layer thickness grows
    toward the surface and ``dp_below != dp_upper``; ``expo`` differs per column
    so the mass weighting and CAPE are genuinely anisotropic.
    """
    t_sfc = jnp.asarray(t_sfc, dtype=jnp.float64)
    ncol = t_sfc.shape[0]
    if expo is None:
        expo = 1.3 + 0.3 * jnp.arange(ncol, dtype=jnp.float64)
    xi = jnp.linspace(0.0, 1.0, nlev + 1)[None, :]
    p_half = 100.0 + (1.0e5 - 100.0) * xi ** expo[:, None]  # monotone, nonuniform
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    frac = jnp.linspace(0.0, 1.0, nlev)[None, :]  # 0 at top, 1 at surface
    T = t_top + (t_sfc[:, None] - t_top) * frac
    q_v = supersat * saturation_mixing_ratio(T, p_full)
    return T, q_v, p_full, p_half


def _convecting():
    """Primary strongly-convecting fixture (distinct warm surfaces)."""
    return _make_columns(jnp.array([300.0, 302.0, 304.0, 306.0]))


def _assert_finite_and_diverges(module_val, counterfactual, *, tol=1e-6):
    """A canary is load-bearing only if the counterfactual is finite AND
    differs materially — ``not allclose`` alone passes trivially on NaN."""
    a = np.asarray(module_val)
    b = np.asarray(counterfactual)
    assert np.all(np.isfinite(a)), "module output not finite"
    assert np.all(np.isfinite(b)), "counterfactual not finite"
    assert float(np.max(np.abs(a - b))) > tol, (
        "counterfactual did not diverge — the perturbed term is not load-bearing"
    )


# ---------------------------------------------------------------------------
# Independent oracle: original-coord bottom-to-top Python sweep (vs the
# module's reversed lax.scan), 2x2 solve written in mass-fraction form.
# ---------------------------------------------------------------------------
def _oracle_sweep(
    T, q_v, p_full, dp, mixing_fraction, sharpness,
    *, latent=True, target="moist", swap_outputs=False, strat=True,
):
    """One bottom-to-top adjustment sweep, assembled independently.

    ``latent`` / ``target`` / ``swap_outputs`` / ``strat`` are perturbation
    switches used ONLY by the canary tests; the base call (all defaults) is the
    faithful oracle.
    """
    ncol, nlev = T.shape
    T_work = T
    q_work = q_v
    # Bottom (index nlev-1, surface) up to the top (index 0); pair (i, i-1).
    for i in range(nlev - 1, 0, -1):
        i_below, i_upper = i, i - 1
        T_below = T_work[:, i_below]
        q_below = q_work[:, i_below]
        p_below = p_full[:, i_below]
        dp_below = dp[:, i_below]
        T_upper = T_work[:, i_upper]
        q_upper = q_work[:, i_upper]
        p_upper = p_full[:, i_upper]
        dp_upper = dp[:, i_upper]

        p_mid = 0.5 * (p_below + p_upper)
        T_mid = 0.5 * (T_below + T_upper)
        dp_pair = jnp.clip(p_below - p_upper, 1.0, None)
        actual_dTdp = (T_below - T_upper) / dp_pair
        gamma_m = moist_adiabat_lapse_rate(T_mid, p_mid)  # GIVEN
        gamma_dry = constants.R_d * T_mid / (constants.c_pd * p_mid)
        gamma_target = gamma_m if target == "moist" else gamma_dry  # canary switch
        instability = (actual_dTdp - gamma_target) / jnp.clip(gamma_dry, 1e-10, None)
        strat_gate = stratosphere_mass_flux_gate(p_mid) if strat else 1.0  # GIVEN
        blend = jax.nn.sigmoid(sharpness * instability) * mixing_fraction * strat_gate

        # 2x2 solve as a mass-fraction distribution about the weighted mean —
        # the unique solution of the lapse + dry-enthalpy equations, written
        # DIFFERENTLY than the module's closed form (non-circular).
        total_dp = dp_below + dp_upper
        mean_T = (dp_below * T_below + dp_upper * T_upper) / total_dp
        step = gamma_target * dp_pair
        T_new_upper = mean_T - (dp_below / total_dp) * step
        T_new_below = mean_T + (dp_upper / total_dp) * step

        T_adj_upper = T_upper + blend * (T_new_upper - T_upper)
        T_adj_below = T_below + blend * (T_new_below - T_below)

        q_sat_upper = saturation_mixing_ratio(T_adj_upper, p_upper)  # GIVEN
        q_sat_below = saturation_mixing_ratio(T_adj_below, p_below)
        q_new_upper = jnp.minimum(q_upper, q_sat_upper)
        q_new_below = jnp.minimum(q_below, q_sat_below)
        q_adj_upper = q_upper + blend * (q_new_upper - q_upper)
        q_adj_below = q_below + blend * (q_new_below - q_below)

        dq_upper_pa = (q_upper - q_adj_upper) * dp_upper
        dq_below_pa = (q_below - q_adj_below) * dp_below
        delta_T_lh = (
            constants.L_v * (dq_upper_pa + dq_below_pa)
            / (constants.c_pd * (dp_below + dp_upper))
        )
        if latent:
            T_adj_upper = T_adj_upper + delta_T_lh
            T_adj_below = T_adj_below + delta_T_lh

        # Level assignment (canary: swap_outputs writes each adjusted value to
        # the WRONG level, keeping pressures/dp physical).
        set_below, set_upper = (
            (T_adj_upper, T_adj_below) if swap_outputs else (T_adj_below, T_adj_upper)
        )
        qb, qu = (
            (q_adj_upper, q_adj_below) if swap_outputs else (q_adj_below, q_adj_upper)
        )
        T_work = T_work.at[:, i_below].set(set_below).at[:, i_upper].set(set_upper)
        q_work = q_work.at[:, i_below].set(qb).at[:, i_upper].set(qu)
    return T_work, q_work


def _oracle_driver(T, q_v, p_full, p_half, dt, config, *, gate=True, **sweep_kw):
    """Full Manabe driver assembled independently from ``_oracle_sweep``."""
    dp = p_half[:, 1:] - p_half[:, :-1]
    T_adj, q_adj = T, q_v
    for _ in range(config.n_iterations):
        T_adj, q_adj = _oracle_sweep(
            T_adj, q_adj, p_full, dp,
            config.mixing_fraction, config.instability_blend_sharpness,
            **sweep_kw,
        )
    cape = compute_cape(T, T_adj, p_full, p_half)  # GIVEN
    if gate:
        cape_gate = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
    else:
        cape_gate = jnp.ones_like(cape)  # canary: no CAPE gate
    dT_dt = cape_gate[:, None] * (T_adj - T) / dt
    dq_v_dt = cape_gate[:, None] * (q_adj - q_v) / dt
    weight = dp / constants.g
    local_cond = jnp.maximum(-dq_v_dt, 0.0)
    col_local_cond = jnp.sum(local_cond * weight, axis=-1, keepdims=True)
    col_net_drying = jnp.clip(
        -jnp.sum(dq_v_dt * weight, axis=-1, keepdims=True), 0.0, None
    )
    dq_c_conv_dt = local_cond * safe_divide(col_net_drying, col_local_cond, eps=1e-20)
    return {
        "dT_dt": dT_dt,
        "dq_v_dt": dq_v_dt,
        "dq_c_conv_dt": dq_c_conv_dt,
        "cape": cape,
        "convective_mask": cape_gate,
    }


def _oracle_T_adj(T, q_v, p_full, p_half, config):
    """Pre-gate adjusted (T, q) via the independent sweep."""
    dp = p_half[:, 1:] - p_half[:, :-1]
    T_adj, q_adj = T, q_v
    for _ in range(config.n_iterations):
        T_adj, q_adj = _oracle_sweep(
            T_adj, q_adj, p_full, dp,
            config.mixing_fraction, config.instability_blend_sharpness,
        )
    return T_adj, q_adj


# ---------------------------------------------------------------------------
# 1. Full-output faithfulness pin (rel 1e-12) vs the independent oracle.
# ---------------------------------------------------------------------------
def test_manabe_full_output_matches_independent_oracle():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    ref = _oracle_driver(T, q_v, p_full, p_half, _DT, cfg)

    # Non-vacuity: the adjustment fired and condensed water.
    assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-4
    assert float(jnp.min(out.dq_v_dt)) < -1e-8
    assert float(jnp.max(out.dq_c_conv_dt)) > 1e-9
    assert float(jnp.max(out.convective_mask)) > 0.5

    for name in ("dT_dt", "dq_v_dt", "dq_c_conv_dt", "cape", "convective_mask"):
        got = np.asarray(getattr(out, name))
        want = np.asarray(ref[name])
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=1e-14,
            err_msg=f"Manabe DCA {name} departs from the independent oracle",
        )


def test_dispatch_matches_manabe_driver():
    """The public ``dca_convection`` default routes to the Manabe driver."""
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    disp = dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    direct = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    for name in ("dT_dt", "dq_v_dt", "dq_c_conv_dt", "cape", "convective_mask"):
        np.testing.assert_array_equal(
            np.asarray(getattr(disp, name)), np.asarray(getattr(direct, name))
        )


# ---------------------------------------------------------------------------
# 2. TRUTH-TIER (oracle-independent): column moist static energy conserved.
#    Sum over levels of (c_pd*dT_dt + L_v*dq_v_dt)*dp/g == 0 to round-off.
#    TIGHTENS the legacy inner-sweep bound (1e-4) to the round-off the
#    enthalpy-conserving 2x2 solve actually achieves on the FULL driver.
# ---------------------------------------------------------------------------
def test_manabe_column_mse_conserved_to_roundoff():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    dp = p_half[:, 1:] - p_half[:, :-1]
    weight = dp / constants.g

    mse_tend = jnp.sum(
        (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt) * weight, axis=-1
    )  # (ncol,) [W/m^2]
    gross = jnp.sum(jnp.abs(constants.c_pd * out.dT_dt) * weight, axis=-1)
    # Non-vacuity: real convective heating AND real condensation in every column.
    assert float(jnp.min(gross)) > 1.0  # W/m^2
    assert float(jnp.max(jnp.min(out.dq_v_dt, axis=-1))) < -1e-9  # every col dries
    max_rel = float(jnp.max(jnp.abs(mse_tend) / jnp.maximum(gross, 1e-30)))
    assert max_rel < 1e-12, (
        f"column MSE residual {max_rel:.3e} — the per-pair 2x2 solve + "
        "uniform latent warming should conserve column MSE to round-off"
    )


# ---------------------------------------------------------------------------
# 3. TRUTH-TIER (oracle-independent): total water conserved (closure).
#    NOTE: the Manabe path only ever DRIES (q_adj = q + blend*(min(q,qsat)-q)
#    <= q), so the shared cloud-rescale factor col_net_drying/col_local_cond is
#    identically 1 here — this pins total-water CLOSURE and cloud non-negativity,
#    NOT the rescale machinery (which is dormant shared code for this scheme).
# ---------------------------------------------------------------------------
def test_manabe_total_water_closed_to_roundoff():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    dp = p_half[:, 1:] - p_half[:, :-1]
    weight = dp / constants.g

    # Cloud source is non-negative everywhere (a property the rescale keeps).
    assert float(jnp.min(out.dq_c_conv_dt)) >= 0.0
    col_vap = jnp.sum(out.dq_v_dt * weight, axis=-1)
    col_cld = jnp.sum(out.dq_c_conv_dt * weight, axis=-1)
    col_local_cond = jnp.sum(jnp.maximum(-out.dq_v_dt, 0.0) * weight, axis=-1)
    residual = col_vap + col_cld  # total water tendency; must vanish
    drying = (col_vap < 0.0) & (col_local_cond > 1e-20)
    assert bool(jnp.all(drying))  # every convecting column net-dries here
    scale = jnp.maximum(jnp.abs(col_cld), 1e-30)
    max_rel = float(jnp.max(jnp.abs(residual) / scale))
    assert max_rel < 1e-12, (
        f"total-water residual {max_rel:.3e} — column vapor + cloud tendency "
        "must vanish (total water conserved)"
    )


# ---------------------------------------------------------------------------
# 4. The inner per-pair identity is EXACT (documents that the legacy 1e-4
#    bound is stale).  Requires REAL condensation so it tests the MOIST closure,
#    not merely dry-enthalpy conservation.
# ---------------------------------------------------------------------------
def test_inner_sweep_column_mse_exact():
    T, q_v, p_full, p_half = _convecting()
    dp = p_half[:, 1:] - p_half[:, :-1]
    T_new, q_new, _ = _adjust_one_iteration(
        T, q_v, p_full, dp,
        mixing_fraction=1.0,
        instability_blend_sharpness=DCAConfig().instability_blend_sharpness,
    )
    weight = dp / constants.g
    # Non-vacuity: real thermal adjustment AND real condensation in every column.
    gross = jnp.sum(constants.c_pd * jnp.abs(T_new - T) * weight, axis=-1)
    assert float(jnp.min(gross)) > 1.0
    assert float(jnp.max(jnp.min(q_new - q_v, axis=-1))) < -1e-5  # every col condenses
    dmse = jnp.sum(
        (constants.c_pd * (T_new - T) + constants.L_v * (q_new - q_v)) * weight,
        axis=-1,
    )
    max_rel = float(jnp.max(jnp.abs(dmse) / jnp.maximum(gross, 1e-30)))
    assert max_rel < 1e-12, (
        f"inner-sweep column MSE residual {max_rel:.3e} should be round-off"
    )


# ---------------------------------------------------------------------------
# 5. Canaries: each assembled term is load-bearing.
# ---------------------------------------------------------------------------
def test_canary_latent_heat_is_present_and_material():
    """Dropping the latent-heat term breaks BOTH the oracle match and the
    (oracle-independent) MSE budget — so the module has it and it matters."""
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    no_lh = _oracle_driver(T, q_v, p_full, p_half, _DT, cfg, latent=False)
    _assert_finite_and_diverges(out.dT_dt, no_lh["dT_dt"])
    # Its MSE budget is biased (dry-static-energy-only): residual large.
    dp = p_half[:, 1:] - p_half[:, :-1]
    weight = dp / constants.g
    res = jnp.sum(
        (constants.c_pd * no_lh["dT_dt"] + constants.L_v * no_lh["dq_v_dt"]) * weight,
        axis=-1,
    )
    gross = jnp.sum(jnp.abs(constants.c_pd * no_lh["dT_dt"]) * weight, axis=-1)
    assert float(jnp.max(jnp.abs(res) / jnp.maximum(gross, 1e-30))) > 1e-3


def test_canary_target_is_moist_adiabat_not_dry():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    dry = _oracle_driver(T, q_v, p_full, p_half, _DT, cfg, target="dry")
    _assert_finite_and_diverges(out.dT_dt, dry["dT_dt"])


def test_canary_level_assignment_not_swapped():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    swapped = _oracle_driver(T, q_v, p_full, p_half, _DT, cfg, swap_outputs=True)
    _assert_finite_and_diverges(out.dT_dt, swapped["dT_dt"])


def test_canary_stratosphere_gate_is_present():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    no_strat = _oracle_driver(T, q_v, p_full, p_half, _DT, cfg, strat=False)
    _assert_finite_and_diverges(out.dT_dt, no_strat["dT_dt"])


def test_canary_cape_gate_is_present():
    T, q_v, p_full, p_half = _convecting()
    # Use a raised threshold so the gate is materially < 1 (else gate==1 and the
    # ungated counterfactual would coincide with the module).
    cfg = DCAConfig(cape_threshold=1.0e4)
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    assert float(jnp.max(out.convective_mask)) < 0.99  # gate genuinely partial
    ungated = _oracle_driver(T, q_v, p_full, p_half, _DT, cfg, gate=False)
    _assert_finite_and_diverges(out.dT_dt, ungated["dT_dt"])


# ---------------------------------------------------------------------------
# 6. The CAPE gate is a per-COLUMN scalar (required for MSE conservation: a
#    per-level gate would break the per-pair budget).  Exercised in the gate's
#    UNSATURATED transition with genuinely varying per-column values.
# ---------------------------------------------------------------------------
def test_cape_gate_is_per_column_scalar():
    # Wide surface-T spread -> wide CAPE spread.  ``cape`` is a config-independent
    # diagnostic (the sweep ignores cape_threshold/cape_sharpness), so probe it,
    # then place the threshold at the CAPE median and set the sharpness to span
    # the range +/-~2 sigmoid-units — the realized gate values are then provably
    # varied and bounded away from {0, 1} (not a saturated always-open gate).
    T, q_v, p_full, p_half = _make_columns(jnp.array([285.0, 295.0, 305.0, 315.0]))
    cape_probe = np.asarray(
        _manabe_dca_convection(T, q_v, p_full, p_half, _DT, DCAConfig()).cape
    )
    thr = float(np.median(cape_probe))
    sharp = 4.0 / max(float(np.ptp(cape_probe)), 1.0)
    cfg = DCAConfig(cape_threshold=thr, cape_sharpness=sharp)
    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    mask = np.asarray(out.convective_mask)
    cape = np.asarray(out.cape)

    # Column variation is real (a broadcast bug / always-open gate can't hide).
    assert float(np.ptp(cape)) > 10.0
    assert float(np.ptp(mask)) > 0.3
    assert np.sum((mask > 0.02) & (mask < 0.98)) >= 2  # unsaturated

    T_adj, q_adj = _oracle_T_adj(T, q_v, p_full, p_half, cfg)
    delta_T = np.asarray(T_adj - T)
    delta_q = np.asarray(q_adj - q_v)
    dT_dt = np.asarray(out.dT_dt)
    dq_v_dt = np.asarray(out.dq_v_dt)
    for c in range(T.shape[0]):
        # dT_dt / (T_adj - T) must equal the single per-column gate at EVERY
        # materially-adjusted level (not just one surviving level).
        sel = np.abs(delta_T[c]) > 1e-6
        assert int(sel.sum()) >= 2
        rT = dT_dt[c][sel] * _DT / delta_T[c][sel]
        assert np.all(np.isfinite(rT))
        np.testing.assert_allclose(rT, mask[c], rtol=1e-9, atol=1e-12)
        # The SAME scalar gates the moisture tendency.
        selq = np.abs(delta_q[c]) > 1e-9
        assert int(selq.sum()) >= 2
        rq = dq_v_dt[c][selq] * _DT / delta_q[c][selq]
        assert np.all(np.isfinite(rq))
        np.testing.assert_allclose(rq, mask[c], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Gate application (TARGET-SIDE): the driver runs the FULL sweep, then
#    MULTIPLIES by the per-column gate.  Witness the module's OWN pre-gate sweep
#    (``_adjust_one_iteration`` iterated ``n_iterations``, exactly the driver's
#    internal loop); then at a SET of cape_threshold probes — the (33, 300) J/kg
#    spec endpoints (DCAConfig ``__param_spec__``), the 100 J/kg default, a
#    fixture-derived transition (~800), and a far 1e6 — pin ``dT_dt == cape_gate *
#    pre_gate_sweep`` (and likewise dq_v) with ``cape_gate`` matched to an
#    INDEPENDENT ``cape_trigger`` mapping.  Every probe excludes a
#    short-circuit-to-zero; the UNSATURATED probes (the transition here, and the
#    in-range thr=200 companion ``test_..._in_range_unsaturated`` where the gate
#    is materially sub-unity) additionally exclude a per-level gate, an arg-swap,
#    and a per-variable (T-only / q-only) or interior-interval gate — which a
#    saturated mask~1 probe cannot distinguish from gate^2.  LIMITATION
#    (acknowledged, not a defect of this test): a black-box VALUE test cannot
#    exclude a branch keyed to an EXACT unprobed threshold (or an arbitrary
#    unprobed sub-interval) over the continuous (33, 300) domain — that needs
#    structural inspection; the probes certify the endpoints, the default, an
#    interior (200) value, and a value above the whole tunable range.
# ---------------------------------------------------------------------------
def test_cape_gate_scales_module_sweep_output():
    T, q_v, p_full, p_half = _convecting()
    dp = p_half[:, 1:] - p_half[:, :-1]
    cfg0 = DCAConfig()
    # MODULE's own pre-gate adjustment (the driver runs exactly this internally).
    T_adj, q_adj = T, q_v
    for _ in range(cfg0.n_iterations):
        T_adj, q_adj, _p = _adjust_one_iteration(
            T_adj, q_adj, p_full, dp,
            cfg0.mixing_fraction, cfg0.instability_blend_sharpness,
        )
    pre_dT = np.asarray((T_adj - T) / _DT)
    pre_dq = np.asarray((q_adj - q_v) / _DT)
    assert float(np.max(np.abs(pre_dT))) > 1e-4  # module sweep material (T)
    assert float(np.max(np.abs(pre_dq))) > 1e-9  # module sweep material (q)

    # ``cape`` from the module's OWN pre-gate T_adj (== driver's internal cape).
    cape = compute_cape(T, T_adj, p_full, p_half)  # GIVEN
    cmax = float(jnp.max(cape))
    transition = (float(jnp.median(cape)), 4.0 / max(float(jnp.ptp(cape)), 1.0))
    # A far 1e6 threshold (~3000x the (33,300) J/kg spec bound) with the sharpness
    # tuned so the warmest column's gate is sigmoid(-14) ~ 8.3e-7 — small yet
    # strictly NONZERO, so mask*pre_dT (~2e-9) stays above the atol floor and a
    # short-circuit-to-zero above the whole tunable range fails the identity.
    strong = (1.0e6, 14.0 / (1.0e6 - cmax))
    # Probe the gate identity across cape_threshold's canonical values: the
    # (33,300) spec endpoints, the 100 default, the fixture transition, and 1e6.
    probes = [(33.0, 0.1), (100.0, 0.1), (300.0, 0.1), transition, strong]
    for thr, sharp in probes:
        out = _manabe_dca_convection(
            T, q_v, p_full, p_half, _DT,
            DCAConfig(cape_threshold=thr, cape_sharpness=sharp),
        )
        m = np.asarray(out.convective_mask)
        # (i) gate mapping pinned INDEPENDENTLY of the driver's own report.
        want_gate = np.asarray(cape_trigger(cape, thr, sharp))
        np.testing.assert_allclose(m, want_gate, rtol=1e-12, atol=0.0)
        assert float(np.min(m)) > 0.0  # nonzero -> a short-circuit is detectable
        # (ii) driver output == gate * module pre-gate output (excludes any
        # threshold-dependent short-circuit AND any per-level gate).
        np.testing.assert_allclose(
            np.asarray(out.dT_dt), m[:, None] * pre_dT, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(out.dq_v_dt), m[:, None] * pre_dq, rtol=1e-9, atol=1e-12
        )
    # The transition config also has genuine per-column variation.
    trans_mask = np.asarray(cape_trigger(cape, *transition))
    assert float(np.ptp(trans_mask)) > 0.3


def test_cape_gate_scales_module_sweep_in_range_unsaturated():
    """In-range (cape_threshold=200, inside the (33,300) J/kg spec) probe with
    UNSATURATED, materially-varied per-column gates.  A saturated mask~1 probe
    cannot distinguish gate from gate^2 or catch a per-variable (T-only/q-only)
    or interior-interval gate defect (e.g. ``if 150<thr<250: dq_v*=0.5``); this
    columns-straddle-200 case makes the levelwise T AND q identity strict."""
    T, q_v, p_full, p_half = _make_columns(
        jnp.array([284.0, 288.0, 292.0, 296.0]), t_top=260.0, supersat=1.02,
    )
    dp = p_half[:, 1:] - p_half[:, :-1]
    cfg = DCAConfig(cape_threshold=200.0, cape_sharpness=0.1)  # 200 in [33, 300]
    T_adj, q_adj = T, q_v
    for _ in range(cfg.n_iterations):
        T_adj, q_adj, _p = _adjust_one_iteration(
            T_adj, q_adj, p_full, dp,
            cfg.mixing_fraction, cfg.instability_blend_sharpness,
        )
    pre_dT = np.asarray((T_adj - T) / _DT)
    pre_dq = np.asarray((q_adj - q_v) / _DT)
    assert float(np.max(np.abs(pre_dT))) > 1e-4  # module sweep material (T)
    assert float(np.max(np.abs(pre_dq))) > 1e-9  # module sweep material (q)

    out = _manabe_dca_convection(T, q_v, p_full, p_half, _DT, cfg)
    m = np.asarray(out.convective_mask)
    cape = compute_cape(T, T_adj, p_full, p_half)  # GIVEN
    # Gate mapping pinned independently, and genuinely UNSATURATED within range.
    np.testing.assert_allclose(
        m, np.asarray(cape_trigger(cape, 200.0, 0.1)), rtol=1e-12, atol=0.0
    )
    assert float(np.ptp(m)) > 0.3
    assert int(np.sum((m > 0.02) & (m < 0.98))) >= 2  # materially sub-unity
    # Levelwise scalar-gate identity for BOTH T and q at an unsaturated, in-range
    # threshold: catches a per-variable or interior-interval gate a mask~1 probe
    # would miss (0.5*mask*pre != mask*pre; mask^2 != mask when mask != 1).
    np.testing.assert_allclose(
        np.asarray(out.dT_dt), m[:, None] * pre_dT, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out.dq_v_dt), m[:, None] * pre_dq, rtol=1e-9, atol=1e-12
    )
    # Cloud-water output at THIS in-range threshold is non-negative and
    # column-conservative too — catches a threshold-specific dq_c corruption
    # (e.g. ``if 150<thr<250: dq_c*=0.5``) the default-threshold closure misses.
    weight = dp / constants.g
    assert float(jnp.min(out.dq_c_conv_dt)) >= 0.0
    col_vap = jnp.sum(out.dq_v_dt * weight, axis=-1)
    col_cld = jnp.sum(out.dq_c_conv_dt * weight, axis=-1)
    col_local_cond = jnp.sum(jnp.maximum(-out.dq_v_dt, 0.0) * weight, axis=-1)
    drying = (col_vap < 0.0) & (col_local_cond > 1e-20)
    assert bool(jnp.any(drying))  # the two supra-200-CAPE columns net-dry
    resid = jnp.where(
        drying, jnp.abs(col_vap + col_cld) / jnp.maximum(jnp.abs(col_cld), 1e-30), 0.0
    )
    assert float(jnp.max(resid)) < 1e-12


# ---------------------------------------------------------------------------
# 8. Differentiability (full driver): jax.grad finite and non-zero.
# ---------------------------------------------------------------------------
def test_manabe_driver_differentiable():
    T, q_v, p_full, p_half = _convecting()
    cfg = DCAConfig()

    def loss(T_in):
        out = _manabe_dca_convection(T_in, q_v, p_full, p_half, _DT, cfg)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_c_conv_dt ** 2)

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# ---------------------------------------------------------------------------
# 9. Config defaults are physics — pin them so a silent default change fails.
# ---------------------------------------------------------------------------
def test_manabe_config_defaults():
    cfg = DCAConfig()
    assert cfg.variant == "manabe"
    assert cfg.n_iterations == 3
    assert cfg.mixing_fraction == 1.0
    assert cfg.cape_threshold == 100.0
    assert cfg.cape_sharpness == 0.1
    assert cfg.instability_blend_sharpness == 10.0
