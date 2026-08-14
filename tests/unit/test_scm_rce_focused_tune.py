"""Direct tests for the focused (cross-category) sub-cloud tuner and its CLI.

Everything here is a guard against a search that is quietly smaller than the
log claims: a mistyped parameter name, an inactive scheme, a mistyped tune
mode.  None of these tests runs a column — the resolution and validation all
happen before the first expensive evaluation, which is itself one of the
properties under test.
"""

from __future__ import annotations

import pytest

import scripts.run.run_scm_rce_campaign as camp
from scripts.run.run_scm_rce_convection_intercomparison import (
    FOCUSED_TUNE_CATEGORIES,
    SUBCLOUD_CONVECTION_INCLUDE,
    SUBCLOUD_MICROPHYSICS_INCLUDE,
    SUBCLOUD_TURBULENCE_INCLUDE,
    TUNE_MODES,
    default_focused_include,
    evaluate_scheme,
)


def test_default_include_is_the_union_of_the_three_active_schemes():
    inc = default_focused_include(
        turbulence="clubb", microphysics="morrison", convection="bechtold")
    assert set(inc) == (
        set(SUBCLOUD_TURBULENCE_INCLUDE["clubb"])
        | set(SUBCLOUD_MICROPHYSICS_INCLUDE["morrison"])
        | set(SUBCLOUD_CONVECTION_INCLUDE["bechtold"])
    )
    # Every name is registry-qualified, i.e. "<scheme_key>.<field>".
    assert all(n.count(".") >= 2 for n in inc)


def test_default_include_omits_a_scheme_with_no_downdraft_knob():
    """sbm/dca/kuo expose no downdraft parameter; the focused set must then be
    turbulence + microphysics only, NOT a name that would raise in the tuner."""
    inc = default_focused_include(
        turbulence="clubb", microphysics="morrison", convection="sbm")
    assert not any(n.startswith("atm.conv.") for n in inc)
    assert set(inc) == (
        set(SUBCLOUD_TURBULENCE_INCLUDE["clubb"])
        | set(SUBCLOUD_MICROPHYSICS_INCLUDE["morrison"])
    )


def test_default_include_is_empty_for_a_turbulence_scheme_with_no_set():
    """louis has no curated sub-cloud set, so the default is only the
    microphysics/convection halves — and an EMPTY focused set must be caught by
    the tuner, not silently searched."""
    inc = default_focused_include(
        turbulence="louis", microphysics="kessler", convection="dca")
    assert inc == ()


def test_every_default_name_resolves_in_the_parameter_registry():
    """A curated list rots silently when a parameter is renamed; this is the
    tripwire.  ``build_trainable_params`` raises on an unknown qualified name."""
    from legoesm.training.param_collector import build_registry

    known = {m.qualified_name for m in build_registry()}
    curated = set()
    for table in (SUBCLOUD_TURBULENCE_INCLUDE, SUBCLOUD_MICROPHYSICS_INCLUDE,
                  SUBCLOUD_CONVECTION_INCLUDE):
        for names in table.values():
            curated.update(names)
    missing = sorted(curated - known)
    assert not missing, f"curated sub-cloud parameters no longer exist: {missing}"


def test_focused_categories_are_real_campaign_categories():
    assert set(FOCUSED_TUNE_CATEGORIES) <= set(camp.CATEGORY_CONFIG_FIELD)


def test_unknown_tune_mode_raises_before_anything_runs():
    with pytest.raises(ValueError, match="unknown tune_mode"):
        evaluate_scheme(
            "bechtold", object(),
            days=0.01, dt=600.0, analysis_days=0.01,
            tune_evals=1, seed=0,
            scm_microphysics_substeps=1, scm_convection_substeps=1,
            surface_wind_m_s=5.0, coriolis_s_inv=0.0,
            large_scale_forcing="none", radiation="gray",
            radiation_update_interval_steps=1,
            tune_mode="focussed",  # sic: a typo must not select the historical arm
        )
    assert set(TUNE_MODES) == {"convection", "focused"}


def test_include_naming_an_inactive_scheme_is_a_hard_error(monkeypatch):
    """The parameter belongs to a real scheme, but not to the one that is
    ACTIVE, so it can move nothing.  That must raise rather than shrink the
    search — and it must raise BEFORE the default column is computed."""
    cfg = camp.make_physics_config(
        radiation="gray", turbulence="clubb",
        microphysics="morrison", convection="bechtold")

    def _boom(*_a, **_k):  # pragma: no cover - must never be reached
        raise AssertionError("a column was evaluated before validation")

    monkeypatch.setattr(camp, "run_cached", _boom)
    with pytest.raises(ValueError, match="matched no parameter"):
        camp.tune_focused_params(
            cfg, object(), {},
            categories=FOCUSED_TUNE_CATEGORIES,
            # Tiedtke's downdraft knob with Bechtold active.
            include=("atm.conv.TiedtkeConfig.downdraft_evap_efficiency",),
            days=0.01, dt=600.0, analysis_days=0.01,
            require_equilibrium=False, require_realism=False,
            tune_evals=1, seed=0,
            equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
        )


def test_empty_include_is_a_hard_error(monkeypatch):
    cfg = camp.make_physics_config(
        radiation="gray", turbulence="clubb",
        microphysics="morrison", convection="bechtold")

    def _boom(*_a, **_k):  # pragma: no cover
        raise AssertionError("a column was evaluated before validation")

    monkeypatch.setattr(camp, "run_cached", _boom)
    with pytest.raises(ValueError, match="empty parameter set"):
        camp.tune_focused_params(
            cfg, object(), {},
            categories=FOCUSED_TUNE_CATEGORIES, include=(),
            days=0.01, dt=600.0, analysis_days=0.01,
            require_equilibrium=False, require_realism=False,
            tune_evals=1, seed=0,
            equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
        )


def test_unknown_objective_raises():
    cfg = camp.make_physics_config(radiation="gray", turbulence="clubb")
    with pytest.raises(ValueError, match="unknown objective"):
        camp.tune_focused_params(
            cfg, object(), {},
            categories=FOCUSED_TUNE_CATEGORIES,
            include=("atm.turb.CLUBBParams.c_K",),
            objective="rmse",
            days=0.01, dt=600.0, analysis_days=0.01,
            require_equilibrium=False, require_realism=False,
            tune_evals=1, seed=0,
            equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
        )
