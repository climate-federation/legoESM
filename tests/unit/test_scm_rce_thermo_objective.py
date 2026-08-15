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

from legoesm.training.scm_rce_metrics import (
    DEFAULT_THERMO_MIN_P_PA,
    relative_humidity_profile,
    score_thermo_jax,
    tropospheric_mass_weights,
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


def test_one_kelvin_everywhere_scores_one_on_the_T_term():
    """The declared tolerance is the unit: a uniform 1 K error is exactly 1."""
    ref, p, w = _ref()
    weights = tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w))
    T_term, _rh, _c = score_thermo_jax(
        ref, jnp.asarray(np.asarray(ref.T_ref) + 1.0),
        jnp.asarray(ref.qv_ref), jnp.asarray(p), trop_weights=weights,
        T_scale_K=1.0)
    assert float(T_term) == pytest.approx(1.0, rel=1e-10)


def test_humidity_term_sees_the_free_troposphere():
    """The defect this objective exists to fix, as a control.

    A 20 % relative humidity error applied ONLY above 500 hPa must move the
    thermo score by far more than it moves an absolute-q_v RMSE, because
    absolute q_v up there is ~1e-4 of its surface value.
    """
    ref, p, w = _ref(40)
    weights = np.asarray(
        tropospheric_mass_weights(jnp.asarray(p), jnp.asarray(w)))
    upper = p < 50_000.0
    qv_pert = np.where(upper, np.asarray(ref.qv_ref) * 1.2, ref.qv_ref)

    _T, rh_term, _c = score_thermo_jax(
        ref, jnp.asarray(ref.T_ref), jnp.asarray(qv_pert), jnp.asarray(p),
        trop_weights=jnp.asarray(weights))
    absolute = float(np.sqrt(np.sum(
        weights * (qv_pert - np.asarray(ref.qv_ref)) ** 2)))
    # In units of each metric's own declared tolerance: 5 % RH, and 1 g/kg.
    assert float(rh_term) > 30.0 * (absolute / 1.0e-3), (
        f"rh_term={float(rh_term):.4g} vs absolute-qv-in-g/kg="
        f"{absolute / 1.0e-3:.4g}; the RH term is supposed to be the one that "
        "can see an upper-tropospheric humidity error")


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


def test_relative_humidity_uses_the_ice_blend_below_freezing():
    """A liquid-only curve would inflate cold-level RH; the blend must not."""
    from legoesm.thermo import (
        saturation_mixing_ratio,
        saturation_mixing_ratio_blend,
    )
    T = jnp.asarray([230.0])
    p = jnp.asarray([30_000.0])
    qv = jnp.asarray([1.0e-4])
    rh = float(relative_humidity_profile(T, qv, p)[0])
    liquid = float(qv[0] / saturation_mixing_ratio(T, p)[0])
    blended = float(qv[0] / saturation_mixing_ratio_blend(T, p)[0])
    assert rh == pytest.approx(blended, rel=1e-12)
    assert rh > liquid, (
        "ice saturation is LOWER than liquid at 230 K, so the blended RH must "
        "exceed the liquid-only RH; got blended=%r liquid=%r" % (rh, liquid))


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
