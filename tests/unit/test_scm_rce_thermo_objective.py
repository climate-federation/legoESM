"""Controls for the temperature-and-humidity tuning objective.

Every test here exists because a mis-wired objective does not error — it
silently optimises something else and the campaign reports a confident ranking
for a quantity nobody minimised.  The decisive ones are:

* condensate must have EXACTLY zero effect on the thermo score;
* the tuner's scalar, the CSV's sort key and the CSV's reported objective must
  be the same number;
* the ``physical`` parameter set must differ from ``aggressive`` in BOTH
  directions, which is the only thing that makes it worth having.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.trainable_params import TrainablePhysicsParams
from legoesm.training.scm_rce_metrics import (
    DEFAULT_THERMO_MIN_P_PA,
    THERMO_HUMIDITY_VARIABLES,
    relative_humidity_profile,
    score_thermo_jax,
    reference_cold_point,
    tropospheric_mass_weights,
    tropospheric_min_pressure,
    weighted_rmse,
)


def _ref(n=20):
    """A synthetic tropical-ish reference: warm moist bottom, cold dry top."""
    p = np.linspace(2_000.0, 100_000.0, n)          # top -> surface
    T = np.linspace(200.0, 300.0, n)
    qv = 2.0e-2 * (p / p[-1]) ** 3
    w = np.full(n, 1.0 / n)
    return SimpleNamespace(
        T_ref=T, qv_ref=qv, qcond_ref=np.full(n, 1.0e-4), mass_weights=w,
    ), p, w


def test_perfect_match_scores_exactly_zero():
    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    T_term, rh_term, combined = score_thermo_jax(
        ref, jnp.asarray(ref.T_ref), jnp.asarray(ref.qv_ref), jnp.asarray(p),
        trop_weights=weights)
    assert float(T_term) == 0.0
    assert float(rh_term) == 0.0
    assert float(combined) == 0.0


def test_condensate_has_exactly_no_effect():
    """The whole point of the objective. A condensate change must not move it."""
    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    T = np.asarray(ref.T_ref) + 0.7
    args = (jnp.asarray(T), jnp.asarray(ref.qv_ref), jnp.asarray(p))
    before = float(score_thermo_jax(ref, *args, trop_weights=weights)[2])
    ref.qcond_ref = ref.qcond_ref * 1_000.0
    after = float(score_thermo_jax(ref, *args, trop_weights=weights)[2])
    assert before == after


def test_masked_levels_cannot_leak_a_nan():
    """A NaN inside the region the mask excludes must contribute exactly zero.
    IEEE says 0 * NaN == NaN, so without an explicit mask one stratospheric
    non-finite level would poison a score that never looked there."""
    weights = jnp.asarray([0.0, 0.0, 0.5, 0.5])
    diff = jnp.asarray([jnp.nan, jnp.inf, 1.0, 1.0])
    assert float(weighted_rmse(diff, weights)) == pytest.approx(1.0, rel=1e-12)


def test_humidity_variable_dispatch_raises_on_a_typo():
    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    with pytest.raises(ValueError, match="unknown humidity variable"):
        score_thermo_jax(
            ref, jnp.asarray(ref.T_ref), jnp.asarray(ref.qv_ref),
            jnp.asarray(p), trop_weights=weights, humidity="specific")


def test_logq_and_rh_select_different_terms():
    """The two humidity variables must actually be different numbers, else the
    selectable objective is decoration."""
    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    qv = np.asarray(ref.qv_ref) * 1.3
    args = (jnp.asarray(ref.T_ref), jnp.asarray(qv), jnp.asarray(p))
    _t1, q_logq, _c1 = score_thermo_jax(
        ref, *args, trop_weights=weights, humidity="logq")
    _t2, q_rh, _c2 = score_thermo_jax(
        ref, *args, trop_weights=weights, humidity="rh")
    assert float(q_logq) != float(q_rh)
    assert set(THERMO_HUMIDITY_VARIABLES) == {"logq", "rh"}


def test_logq_scores_a_uniform_relative_error_uniformly():
    """A 10 % moisture error EVERYWHERE is exactly one tolerance unit, whatever
    the profile shape — the property an absolute q_v RMSE does not have."""
    ref, p, w = _ref(40)
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    qv = np.asarray(ref.qv_ref) * 1.10
    _T, q_term, _c = score_thermo_jax(
        ref, jnp.asarray(ref.T_ref), jnp.asarray(qv), jnp.asarray(p),
        trop_weights=weights, humidity="logq", logq_scale=math.log(1.10))
    assert float(q_term) == pytest.approx(1.0, rel=1e-9)


def _tropopause_column(n=60, cold_z_km=16.0):
    """A column with an UNAMBIGUOUS cold point at ``cold_z_km``."""
    z_km = np.linspace(0.0, 30.0, n)[::-1]              # top -> surface
    p = 100_000.0 * np.exp(-z_km / 8.0)
    T = 200.0 + 100.0 * np.exp(-((z_km - 0.0) / 12.0) ** 2)
    T = np.where(z_km > cold_z_km, 200.0 + 3.0 * (z_km - cold_z_km), T)
    T[np.argmin(np.abs(z_km - cold_z_km))] = 190.0      # the minimum
    return T, p, z_km * 1000.0


def test_tropospheric_min_pressure_sits_below_the_reference_cold_point():
    """The mask must be derived from the reference's own cold point, not from a
    bound that happens to be near it."""
    T, p, z_m = _tropopause_column()
    idx, T_cold, z_cold = reference_cold_point(jnp.asarray(T), jnp.asarray(z_m))
    assert T_cold == pytest.approx(190.0)
    assert 12.0 <= z_cold <= 25.0
    min_p = tropospheric_min_pressure(
        jnp.asarray(T), jnp.asarray(p), jnp.asarray(z_m),
        floor_p_Pa=1.0, buffer_Pa=2_000.0)
    assert min_p == pytest.approx(float(p[idx]) + 2_000.0, rel=1e-9)
    assert min_p > float(p[idx])


def test_cold_point_ignores_a_stratospheric_minimum_above_the_window():
    """The failure an unbounded argmin over `p < 300 hPa` would have: a colder
    level ABOVE the tropopause window (a mesospheric minimum, a noise spike, or
    a profile that keeps cooling to the model top) must not be selected."""
    T, p, z_m = _tropopause_column()
    z_km = z_m / 1000.0
    T = T.copy()
    T[z_km > 26.0] = 150.0            # colder than the real cold point
    _idx, T_cold, z_cold = reference_cold_point(
        jnp.asarray(T), jnp.asarray(z_m))
    assert T_cold == pytest.approx(190.0)
    assert z_cold <= 25.0


def test_cold_point_ignores_a_low_level_inversion():
    """A cold layer BELOW the window must not be selected either."""
    T, p, z_m = _tropopause_column()
    z_km = z_m / 1000.0
    T = T.copy()
    T[z_km < 2.0] = 180.0             # an absurd near-surface minimum
    _idx, T_cold, z_cold = reference_cold_point(
        jnp.asarray(T), jnp.asarray(z_m))
    assert T_cold == pytest.approx(190.0)
    assert z_cold >= 12.0


def test_cold_point_raises_on_a_column_that_does_not_reach_the_window():
    z_m = np.linspace(0.0, 8_000.0, 20)[::-1]
    T = np.linspace(220.0, 300.0, 20)
    with pytest.raises(ValueError, match="no level in"):
        reference_cold_point(jnp.asarray(T), jnp.asarray(z_m))


def test_tropospheric_min_pressure_respects_the_floor():
    T, p, z_m = _tropopause_column()
    min_p = tropospheric_min_pressure(
        jnp.asarray(T), jnp.asarray(p), jnp.asarray(z_m),
        floor_p_Pa=50_000.0, buffer_Pa=0.0)
    assert min_p == 50_000.0


def test_one_kelvin_everywhere_scores_one_on_the_T_term():
    """The declared tolerance is the unit: a uniform 1 K error is exactly 1."""
    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    T_term, _rh, _c = score_thermo_jax(
        ref, jnp.asarray(np.asarray(ref.T_ref) + 1.0),
        jnp.asarray(ref.qv_ref), jnp.asarray(p), trop_weights=weights,
        T_scale_K=1.0)
    assert float(T_term) == pytest.approx(1.0, rel=1e-10)


def test_humidity_term_sees_an_upper_tropospheric_error_analytically():
    """The defect this objective exists to fix, checked against an EXACT value.

    A 20 % moisture error applied to the upper HALF of the column mass (and
    nothing else) must give a ``logq`` term of exactly
    ``sqrt(0.5) * ln(1.2) / logq_scale`` — no tolerance-free "factor of 30"
    assertion, and the number is derivable by hand.  The synthetic profile here
    spans a factor ``(p_top/p_sfc)^3`` in q_v, NOT the reference's own range;
    the claim about the real CRM profile is the probe's, not this test's.
    """
    ref, p, w = _ref(40)
    weights = np.asarray(
        tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w)))
    # Exactly half the (uniform) weights, chosen by index so the expected value
    # is exact rather than approximately half.
    upper = np.zeros(p.size, dtype=bool)
    upper[: p.size // 2] = True
    qv_pert = np.where(upper, np.asarray(ref.qv_ref) * 1.2, ref.qv_ref)

    logq_scale = 0.10
    _T, q_term, _c = score_thermo_jax(
        ref, jnp.asarray(ref.T_ref), jnp.asarray(qv_pert), jnp.asarray(p),
        trop_weights=jnp.asarray(weights), humidity="logq",
        logq_scale=logq_scale)
    mass_fraction = float(np.sum(weights[upper]))
    expected = math.sqrt(mass_fraction) * math.log(1.2) / logq_scale
    assert float(q_term) == pytest.approx(expected, rel=1e-9)


def test_logq_is_blind_to_where_in_the_column_the_error_is():
    """The property that makes `logq` the right variable, stated as an identity
    rather than as a threshold: the SAME fractional error costs the SAME amount
    wherever it is applied, while an absolute q_v RMSE over the identical
    perturbation differs by orders of magnitude between a dry level and a moist
    one.  No invented factor: the logq terms must be EQUAL and the absolute
    ones must differ by the ratio of the q_v values.
    """
    ref, p, w = _ref(40)
    weights = np.asarray(
        tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w)))
    live = np.flatnonzero(weights > 0.0)
    dry, moist = live[:4], live[-4:]          # top-most vs bottom-most levels

    def _terms(idx):
        qv = np.asarray(ref.qv_ref).copy()
        qv[idx] *= 1.2
        _T, q_term, _c = score_thermo_jax(
            ref, jnp.asarray(ref.T_ref), jnp.asarray(qv), jnp.asarray(p),
            trop_weights=jnp.asarray(weights), humidity="logq")
        absolute = float(np.sqrt(np.sum(
            weights * (qv - np.asarray(ref.qv_ref)) ** 2)))
        return float(q_term), absolute

    q_dry, abs_dry = _terms(dry)
    q_moist, abs_moist = _terms(moist)
    # Equal weights per level in this fixture, so the logq cost is identical.
    assert q_dry == pytest.approx(q_moist, rel=1e-9)
    # The absolute metric is not: it scales with q_v itself, which here spans
    # three orders of magnitude between the two groups.
    assert abs_moist > 100.0 * abs_dry


def test_tropospheric_mask_excludes_the_stratosphere():
    ref, p, w = _ref(40)
    weights = np.asarray(
        tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w),
                                  min_p_Pa=DEFAULT_THERMO_MIN_P_PA))
    assert np.all(weights[p < DEFAULT_THERMO_MIN_P_PA] == 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0, rel=1e-12)


def test_empty_mask_raises_rather_than_scoring_every_column_alike():
    ref, p, w = _ref()
    with pytest.raises(ValueError, match="no level at or above"):
        tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w),
                                  min_p_Pa=1.0e9)


def test_relative_humidity_uses_ice_saturation_below_the_blend_window():
    """Independent check, not a call to the same helper: at 230 K the blend is
    fully on the ICE branch, so RH must equal q_v/q_sat_ice computed here."""
    from legoesm.thermo import (
        saturation_mixing_ratio,
        saturation_mixing_ratio_ice,
    )
    T = jnp.asarray([230.0])
    p = jnp.asarray([30_000.0])
    qv = jnp.asarray([1.0e-4])
    rh = float(relative_humidity_profile(T, qv, p)[0])
    ice = float(qv[0] / saturation_mixing_ratio_ice(T, p)[0])
    liquid = float(qv[0] / saturation_mixing_ratio(T, p)[0])
    assert rh == pytest.approx(ice, rel=1e-10)
    assert rh > liquid, (
        f"ice saturation is LOWER than liquid at 230 K, so the RH must exceed "
        f"the liquid-only value; got {rh!r} vs {liquid!r}")


def test_reference_and_model_humidity_go_through_the_same_curve(monkeypatch):
    """Both sides must use ONE saturation curve. Counting the calls is the only
    way to establish it: a version that used liquid for the model and the blend
    for the reference would still produce plausible numbers."""
    from legoesm.training import scm_rce_metrics as M

    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    calls = []
    import legoesm.thermo as thermo

    real = thermo.saturation_mixing_ratio_blend

    def _spy(T, pp, *a, **kw):
        calls.append(float(jnp.asarray(T).mean()))
        return real(T, pp, *a, **kw)

    monkeypatch.setattr(thermo, "saturation_mixing_ratio_blend", _spy)
    M.score_thermo_jax(
        ref, jnp.asarray(np.asarray(ref.T_ref) + 1.0),
        jnp.asarray(ref.qv_ref), jnp.asarray(p),
        trop_weights=weights, humidity="rh")
    assert len(calls) == 2, (
        f"expected exactly two saturation calls (model and reference), got "
        f"{len(calls)}")


# --------------------------------------------------------------------------- #
# Wiring: the tuner's scalar, the CSV sort key and the reported objective
# --------------------------------------------------------------------------- #
def _diag(**kw):
    from scripts.run.run_scm_rce_campaign import RunDiagnostics

    base = dict(
        label="x", config={}, status="ok", reason="",
        T_rmse=1.0, qv_rmse=1.0, cloud_rmse=9.0, precip_rmse=0.0, score=5.0,
        drift_T_rmse_K=0.0, drift_qv_rmse=0.0, drift_qcond_rmse=0.0,
    )
    base.update(kw)
    return RunDiagnostics(**base)


def test_objective_value_selects_the_named_field():
    from scripts.run import run_scm_rce_campaign as camp

    run = _diag(score=5.0, subcloud_score=2.0, thermo_score=0.25)
    assert camp.objective_value(run, "combined") == 5.0
    assert camp.objective_value(run, "subcloud") == 2.0
    assert camp.objective_value(run, "thermo") == 0.25
    assert set(camp.OBJECTIVE_FIELD) == set(camp.TUNE_OBJECTIVES)


def test_objective_value_maps_nan_to_inf_for_every_objective():
    from scripts.run import run_scm_rce_campaign as camp

    run = _diag(score=float("nan"), subcloud_score=float("nan"),
                thermo_score=float("nan"))
    for objective in camp.TUNE_OBJECTIVES:
        assert camp.objective_value(run, objective) == float("inf")


def test_unknown_objective_raises():
    from scripts.run import run_scm_rce_campaign as camp

    with pytest.raises(ValueError, match="unknown objective"):
        camp.objective_value(_diag(), "condensate")


def test_csv_sorts_and_labels_by_the_objective_that_was_tuned(tmp_path):
    """The failure this replaces: a thermo-tuned table sorted by the combined
    score, so the row order and the winner came from a metric nobody minimised.
    """
    import csv as _csv

    from scripts.run import run_scm_rce_convection_intercomparison as driver

    def _res(scheme, combined, thermo):
        return driver.SchemeResult(
            scheme=scheme,
            prior=_diag(score=combined * 2, thermo_score=thermo * 2),
            tuned=_diag(score=combined, thermo_score=thermo),
            records=[],
            subsidence_solve="implicit_flux",
            subsidence_solve_status="forced:x=implicit_flux",
            signature={"objective": "thermo", "tune_evals": 48},
        )

    # `good` wins on thermo and LOSES on the combined score, so the two orders
    # are different and the test can fail.
    results = [_res("good", combined=9.0, thermo=0.10),
               _res("bad", combined=1.0, thermo=0.90)]
    out = tmp_path / "r.csv"
    driver.write_csv(out, results)
    rows = list(_csv.DictReader(out.open()))
    assert [r["scheme"] for r in rows] == ["good", "bad"]
    assert [r["objective"] for r in rows] == ["thermo", "thermo"]
    assert float(rows[0]["tuned_objective"]) == pytest.approx(0.10)
    assert float(rows[0]["objective_improvement_pct"]) == pytest.approx(50.0)


def test_unstamped_checkpoint_is_no_longer_grandfathered():
    from scripts.run import run_scm_rce_convection_intercomparison as driver

    sig = {"days": 100.0}
    assert not driver._signature_compatible({}, sig)
    assert driver._signature_compatible(sig, sig)


def test_every_signature_field_resolves():
    """A field named in _SIGNATURE_FIELDS that is neither an argparse dest nor a
    declared objective constant must raise, not vanish from the signature."""
    from scripts.run import run_scm_rce_convection_intercomparison as driver

    args = driver.build_parser().parse_args(["--schemes", "dca"])
    sig = driver._run_signature(args)
    assert set(sig) == set(driver._SIGNATURE_FIELDS)
    for key in ("param_set", "tune_refine_frac", "thermo_T_scale_K",
                "thermo_rh_scale", "thermo_min_p_Pa",
                "thermo_rh_blend_width_K"):
        assert key in sig


def test_signature_rejects_a_field_nothing_can_supply(monkeypatch):
    """The guard must FAIL when a field is added to _SIGNATURE_FIELDS and to
    nothing else — otherwise it would silently drop from the signature and stop
    protecting anything."""
    from scripts.run import run_scm_rce_convection_intercomparison as driver

    monkeypatch.setattr(
        driver, "_SIGNATURE_FIELDS",
        driver._SIGNATURE_FIELDS + ("a_field_that_does_not_exist",))
    args = driver.build_parser().parse_args(["--schemes", "dca"])
    with pytest.raises(AttributeError, match="a_field_that_does_not_exist"):
        driver._run_signature(args)


# --------------------------------------------------------------------------- #
# The `physical` parameter set
# --------------------------------------------------------------------------- #
def test_physical_set_differs_from_aggressive_in_both_directions():
    from scripts.run import run_scm_rce_campaign as camp

    tier, include_tier0, exclude = camp.resolve_param_selection(
        "atm.conv.BechtoldConfig", "physical")
    assert tier == "aggressive"
    # ADDS the AD-unreachable CAPE trigger ...
    assert "atm.conv.BechtoldConfig.cape_threshold" in include_tier0
    # ... and DROPS the numerics knob.
    assert "atm.conv.BechtoldConfig.mc_normalize_scale" in exclude


def test_physical_set_never_admits_a_genuine_numerics_tier0():
    """Emanuel's tier-0 downdraft gates are smoothing widths and divisor floors,
    not AD-unreachable physics; admitting them would be tuning the numerics."""
    from scripts.run import run_scm_rce_campaign as camp

    _tier, include_tier0, _exclude = camp.resolve_param_selection(
        "atm.conv.EmanuelConfig", "physical")
    assert "atm.conv.EmanuelConfig.cape_threshold" in include_tier0
    for name in ("downdraft_dhdp_min", "downdraft_ep_gate_threshold",
                 "downdraft_ep_gate_width", "downdraft_freeze_transition_K"):
        assert f"atm.conv.EmanuelConfig.{name}" not in include_tier0


def test_physical_set_builds_and_is_larger_than_extended():
    from legoesm.training.param_collector import build_trainable_params
    from scripts.run import run_scm_rce_campaign as camp

    key = "atm.conv.BechtoldConfig"
    tier, include_tier0, exclude = camp.resolve_param_selection(key, "physical")
    physical = build_trainable_params(
        active_scheme_keys={key}, tier=tier, include_tier0=include_tier0,
        exclude=exclude, dtype=jnp.float64)
    extended = build_trainable_params(
        active_scheme_keys={key}, tier="extended", dtype=jnp.float64)
    names = {c.name for c in physical.constraints}
    assert len(names) > len(extended.constraints)
    assert f"{key}.cape_threshold" in names
    assert f"{key}.mc_normalize_scale" not in names


def test_unknown_param_set_raises():
    from scripts.run import run_scm_rce_campaign as camp

    with pytest.raises(ValueError, match="unknown param_set"):
        camp.resolve_param_selection("atm.conv.DCAConfig", "everything")


def test_tier0_optin_is_refused_for_a_non_tier0_name():
    from legoesm.training.param_collector import build_trainable_params

    with pytest.raises(ValueError, match="NOT tier 0"):
        build_trainable_params(
            active_scheme_keys={"atm.conv.BechtoldConfig"},
            tier="extended",
            include_tier0=("atm.conv.BechtoldConfig.rprcon",),
            dtype=jnp.float64)


def test_tier0_stays_excluded_without_the_optin():
    from legoesm.training.param_collector import build_trainable_params

    params = build_trainable_params(
        active_scheme_keys={"atm.conv.BechtoldConfig"}, tier="aggressive",
        dtype=jnp.float64)
    assert "atm.conv.BechtoldConfig.cape_threshold" not in {
        c.name for c in params.constraints}


# --------------------------------------------------------------------------- #
# The local refinement
# --------------------------------------------------------------------------- #
def test_frac_from_value_inverts_interp_on_both_scales():
    from scripts.run import run_scm_rce_campaign as camp

    linear = SimpleNamespace(min_val=0.0, max_val=4.0, transform="sigmoid")
    log = SimpleNamespace(min_val=1.0e-6, max_val=1.0e-2, transform="sigmoid")
    for c in (linear, log):
        for frac in (0.0, 0.13, 0.5, 0.87, 1.0):
            value = camp._interp(c, frac)
            assert camp._frac_from_value(c, value) == pytest.approx(
                frac, abs=1e-12)


def test_frac_from_value_clamps_a_default_outside_its_own_bounds():
    from scripts.run import run_scm_rce_campaign as camp

    c = SimpleNamespace(min_val=1.0, max_val=2.0, transform="sigmoid")
    assert camp._frac_from_value(c, 0.5) == 0.0
    assert camp._frac_from_value(c, 9.0) == 1.0
    assert math.isfinite(camp._frac_from_value(c, 1.5))


def test_tuner_minimises_the_named_objective_and_the_csv_agrees(tmp_path,
                                                                monkeypatch):
    """The decisive wiring test: run the REAL tuner against a stub evaluator in
    which `combined` and `thermo` prefer OPPOSITE candidates, and require the
    selected configuration, the record scores and the CSV to follow whichever
    was asked for.  A dispatch-dictionary test cannot establish this.
    """
    import csv as _csv

    from scripts.run import run_scm_rce_campaign as camp
    from scripts.run import run_scm_rce_convection_intercomparison as driver

    base = camp.make_physics_config(convection="mass_flux")

    # Key the stub on ONE named parameter crossing the MIDPOINT of its own
    # declared range.  The tuner's first phase sweeps every parameter to the
    # 25 % and 75 % points, so a candidate on each side is GUARANTEED to be
    # offered — the test cannot fail for want of a lucky draw.
    from legoesm.training.param_collector import build_trainable_params

    _c0, _s0, sub0 = camp._active_subconfig(base, "convection")
    key = camp._scheme_key_for_subconfig(sub0)
    tier, include_tier0, exclude = camp.resolve_param_selection(key, "physical")
    params = build_trainable_params(
        active_scheme_keys={key}, tier=tier, include_tier0=include_tier0,
        exclude=exclude, dtype=jnp.float64)
    probe = params.constraints[0]
    midpoint = 0.5 * (float(probe.min_val) + float(probe.max_val))

    def _probe_value(cfg) -> float:
        _c, _s, sub = camp._active_subconfig(cfg, "convection")
        return float(getattr(camp._tunable_subconfig(sub), probe.field))

    def _fake_run_cached(cache, cfg, ref, *, label, **kw):
        # Above the midpoint is good on `thermo` and bad on `combined`; below,
        # the reverse.  The two objectives therefore have opposite optima.
        if _probe_value(cfg) > midpoint:
            return _diag(score=9.0, thermo_score=0.1, subcloud_score=1.0)
        return _diag(score=1.0, thermo_score=0.9, subcloud_score=1.0)

    monkeypatch.setattr(camp, "run_cached", _fake_run_cached)
    common = dict(
        days=0.01, dt=600.0, analysis_days=0.01,
        require_equilibrium=False, require_realism=False,
        equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
    )
    picked = {}
    for objective in ("combined", "thermo"):
        cfg, records, tuned, _stats = camp.tune_category_winner(
            "convection", base, object(), {}, tune_evals=32, seed=7,
            objective=objective, param_set="physical", **common)
        picked[objective] = (_probe_value(cfg), tuned, records)

    combined_value, combined_run, _cr = picked["combined"]
    thermo_value, thermo_run, thermo_records = picked["thermo"]
    assert combined_run.score == 1.0 and combined_run.thermo_score == 0.9
    assert thermo_run.thermo_score == 0.1 and thermo_run.score == 9.0
    assert thermo_value > midpoint >= combined_value, (
        "the two objectives must select configurations on OPPOSITE sides of "
        f"{probe.field}'s midpoint, else the test cannot fail "
        f"(combined={combined_value}, thermo={thermo_value}, mid={midpoint})")
    # The RECORDS carry the objective minimised, not the combined score.
    assert thermo_records and thermo_records[0].score_tuned == 0.1

    res = driver.SchemeResult(
        scheme="mass_flux", prior=_diag(score=1.0, thermo_score=0.9),
        tuned=thermo_run, records=list(thermo_records),
        signature={"objective": "thermo", "tune_evals": 24},
    )
    out = tmp_path / "one.csv"
    driver.write_csv(out, [res])
    row = next(iter(_csv.DictReader(out.open())))
    assert row["objective"] == "thermo"
    assert float(row["tuned_objective"]) == pytest.approx(0.1)
    assert float(row["tuned_score"]) == pytest.approx(9.0)


def test_physical_set_pins_the_admitted_tier0_parameters_for_every_scheme():
    """The tier-0 opt-in is selected by matching a free-text reference field, so
    the RESULT is pinned here: exactly the CAPE trigger of the eight schemes
    that declare one, and nothing else.  A reworded reference string, or a
    numerics parameter acquiring the marker, goes red."""
    from scripts.run import run_scm_rce_campaign as camp

    expected = {
        "atm.conv.SBMConfig": {"atm.conv.SBMConfig.cape_threshold"},
        "atm.conv.DCAConfig": {"atm.conv.DCAConfig.cape_threshold"},
        "atm.conv.KuoConfig": set(),
        "atm.conv.MassFluxConfig": {"atm.conv.MassFluxConfig.cape_threshold"},
        "atm.conv.ConvectiveEDMFConfig": {
            "atm.conv.ConvectiveEDMFConfig.cape_threshold"},
        "atm.conv.ZhangMcFarlaneConfig": {
            "atm.conv.ZhangMcFarlaneConfig.cape_threshold"},
        "atm.conv.KainFritschConfig": set(),
        "atm.conv.EmanuelConfig": {"atm.conv.EmanuelConfig.cape_threshold"},
        "atm.conv.TiedtkeConfig": {"atm.conv.TiedtkeConfig.cape_threshold"},
        "atm.conv.BechtoldConfig": {"atm.conv.BechtoldConfig.cape_threshold"},
    }
    for key, want in expected.items():
        _tier, include_tier0, _exclude = camp.resolve_param_selection(
            key, "physical")
        assert set(include_tier0) == want, f"{key}: {include_tier0}"


def test_physical_set_defaults_are_all_inside_their_bounds():
    """A default outside its own bounds is silently clamped by the sigmoid
    seeding, so candidate zero would not be the a-priori configuration. Check
    every scheme, since the tuner only raises for the one being run."""
    from legoesm.training.param_collector import build_trainable_params
    from scripts.run import run_scm_rce_campaign as camp

    offenders = {}
    for scheme in ("sbm", "dca", "kuo", "mass_flux", "edmf",
                   "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke",
                   "bechtold"):
        cfg = camp.make_physics_config(convection=scheme)
        _c, _s, sub = camp._active_subconfig(cfg, "convection")
        key = camp._scheme_key_for_subconfig(sub)
        if key is None:
            continue
        tier, include_tier0, exclude = camp.resolve_param_selection(
            key, "physical")
        params = build_trainable_params(
            active_scheme_keys={key}, tier=tier, include_tier0=include_tier0,
            exclude=exclude, dtype=jnp.float64)
        values = params.as_dict()
        for c in params.constraints:
            v = float(values[c.name])
            if not (float(c.min_val) <= v <= float(c.max_val)):
                offenders[c.name] = (v, float(c.min_val), float(c.max_val))
    assert not offenders, offenders


def test_run_cached_builds_a_key_for_every_argument_it_is_given(monkeypatch):
    """The smoke caught this and no unit test would have: adding a parameter to
    `run_cached` without widening `_config_cache_key` raises only once a real
    evaluation is attempted, minutes into a job.  Exercise the real key
    construction with the full argument set the driver passes."""
    from scripts.run import run_scm_rce_campaign as camp

    calls = []

    def _stub_run_scm_rce(cfg, ref, **kw):
        calls.append(kw)
        return _diag()

    monkeypatch.setattr(camp, "run_scm_rce", _stub_run_scm_rce)
    cfg = camp.make_physics_config(convection="dca")
    common = dict(
        label="x", days=0.01, dt=600.0, analysis_days=0.01,
        require_equilibrium=False, require_realism=False,
        equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
        scm_microphysics_substeps=1, scm_convection_substeps=1,
        surface_wind_m_s=5.0, coriolis_s_inv=0.0,
        large_scale_forcing="none", bl_anchor_top_m=-1.0,
        subcloud_top_m=1000.0,
    )
    cache: dict = {}
    camp.run_cached(cache, cfg, object(), thermo_humidity="logq", **common)
    camp.run_cached(cache, cfg, object(), thermo_humidity="rh", **common)
    assert len(cache) == 2, (
        "the humidity variable must be part of the cache key: it decides which "
        "term becomes thermo_score, so one entry cannot serve both")
    assert calls[0]["thermo_humidity"] == "logq"
    assert calls[1]["thermo_humidity"] == "rh"


def test_tune_stats_count_only_this_calls_integrations(monkeypatch):
    """`unique_evals` must be columns THIS call integrated, not the cache size.

    Exercised through the REAL `run_cached`, with only the integration stubbed,
    so the caching and the counting are the production ones: a stub that
    replaces `run_cached` would never insert anything and the accounting could
    report any value without failing.  The cache is pre-populated to stand in
    for a caller that shares one across schemes.
    """
    from scripts.run import run_scm_rce_campaign as camp

    integrations = []

    def _stub_run_scm_rce(cfg, ref, **kw):
        integrations.append(kw["label"])
        return _diag(score=1.0, thermo_score=1.0, subcloud_score=1.0)

    monkeypatch.setattr(camp, "run_scm_rce", _stub_run_scm_rce)
    base = camp.make_physics_config(convection="dca")
    cache = {"a-key-from-another-scheme": _diag(), "and-another": _diag()}
    common = dict(
        days=0.01, dt=600.0, analysis_days=0.01,
        require_equilibrium=False, require_realism=False,
        equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
    )
    _cfg, records, _tuned, stats = camp.tune_category_winner(
        "convection", base, object(), cache, tune_evals=12, seed=3,
        objective="thermo", param_set="physical", refine_frac=0.5, **common)

    assert len(records) == 1, "dca exposes exactly its CAPE trigger here"
    assert stats["requested_evals"] == 12
    assert stats["random_proposals"] + stats["refine_proposals"] <= 12
    # The two foreign entries must not be counted, and the default run (made
    # BEFORE the loop) must not be either.
    assert stats["unique_evals"] == len(cache) - 3, (
        f"unique_evals={stats['unique_evals']} vs cache={len(cache)}; it must "
        "count only the keys this call inserted after the default")
    assert stats["unique_evals"] <= len(integrations)


def test_the_default_candidate_is_not_re_evaluated(monkeypatch):
    """MEASURED defect: `_candidate_values` yields the defaults first, and the
    round trip through the sigmoid raw space is accurate only to ~1 ULP, so
    "candidate zero" was a bit-perturbed configuration with its own cache key
    and its own 100-day integration. On a chaotic column that scored 0.24 away
    from the a-priori run, and the tuner banked it as an improvement with zero
    parameters moved. The defaults must never be integrated twice.
    """
    from scripts.run import run_scm_rce_campaign as camp

    labels = []

    def _stub_run_scm_rce(cfg, ref, **kw):
        labels.append(kw["label"])
        return _diag(score=1.0, thermo_score=1.0, subcloud_score=1.0)

    monkeypatch.setattr(camp, "run_scm_rce", _stub_run_scm_rce)
    base = camp.make_physics_config(convection="dca")
    _cfg, _records, _tuned, stats = camp.tune_category_winner(
        "convection", base, object(), {}, tune_evals=6, seed=1,
        objective="thermo", param_set="physical",
        days=0.01, dt=600.0, analysis_days=0.01,
        require_equilibrium=False, require_realism=False,
        equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0)

    assert labels, "the tuner integrated nothing"
    assert labels[0].startswith("tune-default:"), labels[0]
    # eval000 IS the defaults, so it must not reach the integrator at all.
    assert not any(l.endswith("eval000") for l in labels), (
        f"the default candidate was integrated a second time: {labels}")
    assert stats["unique_evals"] >= 1


def test_the_collectors_defaults_are_not_the_shipped_defaults():
    """The mechanism behind the default-candidate skip, compared against the
    RIGHT pair.

    ``build_trainable_params`` seeds its raw space FROM the shipped NamedTuple
    default, so ``as_dict()`` already returns a value that has been through
    ``value -> sigmoid raw -> value``. Comparing ``as_dict()`` against a second
    round trip of itself is circular and shows nothing — two earlier versions of
    this test did exactly that and "passed"/"failed" for the wrong reason. The
    comparison that matters is the collector's default against the value the
    scheme actually ships, which is what the probe measured: sbm's tau_c ships
    as 7200.0 and comes back as 7199.999999999998.
    """
    from legoesm.training.param_collector import build_trainable_params
    from scripts.run import run_scm_rce_campaign as camp

    inexact = {}
    for scheme in ("sbm", "mass_flux", "emanuel"):
        cfg = camp.make_physics_config(convection=scheme)
        _c, _s, sub = camp._active_subconfig(cfg, "convection")
        key = camp._scheme_key_for_subconfig(sub)
        tier, include_tier0, exclude = camp.resolve_param_selection(
            key, "physical")
        params = build_trainable_params(
            active_scheme_keys={key}, tier=tier, include_tier0=include_tier0,
            exclude=exclude, dtype=jnp.float64)
        collected = params.as_dict()
        tunable = camp._tunable_subconfig(sub)
        inexact[scheme] = [
            c.field for c in params.constraints
            if float(collected[c.name]) != float(getattr(tunable, c.field))
        ]
    assert all(inexact.values()), (
        "the collector now reproduces the shipped defaults exactly; the "
        "default-candidate skip can be revisited, but only with a measurement "
        f"showing the two columns score the same. inexact per scheme: {inexact}")


def test_a_default_candidate_win_is_flagged_in_the_table(tmp_path):
    """Checkpoints written before the skip landed can carry an "improvement"
    that is only the chaotic gap between two runs of the same configuration.
    The table must say so without the reader having to know the history."""
    import csv as _csv

    from scripts.run import run_scm_rce_convection_intercomparison as driver

    fake = driver.SchemeResult(
        scheme="emanuel",
        prior=_diag(score=1.0, thermo_score=6.961),
        tuned=_diag(score=1.0, thermo_score=6.721,
                    label="tune:convection:emanuel:eval000"),
        records=[], signature={"objective": "thermo", "tune_evals": 240},
    )
    real = driver.SchemeResult(
        scheme="edmf",
        prior=_diag(score=1.0, thermo_score=7.879),
        tuned=_diag(score=1.0, thermo_score=3.234,
                    label="tune:convection:edmf:eval037"),
        records=[], signature={"objective": "thermo", "tune_evals": 72},
    )
    out = tmp_path / "r.csv"
    driver.write_csv(out, [fake, real])
    rows = {r["scheme"]: r for r in _csv.DictReader(out.open())}
    assert rows["emanuel"]["tuned_is_default_candidate"] == "True"
    assert rows["edmf"]["tuned_is_default_candidate"] == "False"
    assert rows["emanuel"]["tuned_winner_label"].endswith("eval000")
