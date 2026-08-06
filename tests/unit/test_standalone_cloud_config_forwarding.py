"""The MPAS lane's cloud-config builder must forward every tuned cloud scalar.

``_standalone_cloud_config`` (MPAS / lean-loop path) is documented as mirroring
the FV pipeline's ``build_cloud_config`` call.  It did NOT: five kwargs were
missing, so ``--cloud-optics-inhomogeneity``, ``--cloud-inhomogeneity-factor``,
``--cloud-fsd``, ``--cloud-diagnostic-condensate-scheme`` and
``--cloud-adiabatic-lwc-rate`` were accepted by the CLI and then silently
dropped on the production MPAS lane — a run would use the scheme defaults no
matter what the user passed (codex review 2026-07-30).

Each assertion below uses a NON-DEFAULT value, so it fails if the corresponding
forward is deleted (the gate: a test that inspects wiring must fail when the
feature is removed).
"""
from types import SimpleNamespace

import pytest

from legoesm.driver.model_driver import _standalone_cloud_config
from legoesm.atmosphere.physics.clouds.config import build_cloud_config


def _cfg(**over):
    """An ExperimentConfig-like object carrying non-default cloud scalars."""
    base = dict(
        convective_cloud=False,
        cloud_rh_crit=0.85,
        cloud_q_c_diagnostic=1.0e-4,
        cloud_conv_cloud_max=None,
        cloud_conv_cloud_condensate=None,
        cloud_inhomogeneity_factor=0.7,          # non-default (default 1.0)
        cloud_optics_inhomogeneity="two_region",  # non-default ("constant")
        cloud_fsd=0.5,                            # non-default (0.75)
        cloud_diagnostic_condensate_scheme="adiabatic",   # non-default
        cloud_adiabatic_lwc_rate=2.0e-6,          # non-default
        cloud_p_xr=None,
        cloud_alpha_xr=None,
        cloud_clubb_cf_override_strength=None,
        cloud_clubb_cf_override_floor=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_inhomogeneity_and_condensate_selectors_reach_the_cloud_config():
    cc = _standalone_cloud_config(_cfg(), "sundqvist")
    assert cc is not None
    # The five that were silently dropped.
    assert cc.cloud_optics_inhomogeneity == "two_region"
    assert cc.cloud_inhomogeneity_factor == pytest.approx(0.7)
    assert cc.cloud_fsd == pytest.approx(0.5)
    assert cc.diagnostic_condensate_scheme == "adiabatic"
    assert cc.adiabatic_lwc_rate == pytest.approx(2.0e-6)
    # The ones that already worked must keep working.
    assert cc.rh_crit == pytest.approx(0.85)
    assert cc.q_c_diagnostic == pytest.approx(1.0e-4)


def test_matches_a_direct_build_cloud_config_call():
    """The MPAS helper and a direct build must agree field-for-field, which is
    what "mirrors the FV pipeline's call" is supposed to mean."""
    cfg = _cfg()
    got = _standalone_cloud_config(cfg, "sundqvist")
    want = build_cloud_config(
        "sundqvist",
        convective_cloud=False,
        rh_crit=cfg.cloud_rh_crit,
        q_c_diagnostic=cfg.cloud_q_c_diagnostic,
        cloud_inhomogeneity_factor=cfg.cloud_inhomogeneity_factor,
        cloud_optics_inhomogeneity=cfg.cloud_optics_inhomogeneity,
        cloud_fsd=cfg.cloud_fsd,
        diagnostic_condensate_scheme=cfg.cloud_diagnostic_condensate_scheme,
        adiabatic_lwc_rate=cfg.cloud_adiabatic_lwc_rate,
    )
    assert got == want


def test_scheme_none_still_returns_none():
    assert _standalone_cloud_config(_cfg(), "none") is None


def test_absent_fields_fall_back_to_scheme_defaults():
    """A config object lacking the new fields must not raise (getattr default)."""
    bare = SimpleNamespace(convective_cloud=False)
    cc = _standalone_cloud_config(bare, "sundqvist")
    assert cc is not None
    assert cc.cloud_optics_inhomogeneity == "constant"
    assert cc.cloud_inhomogeneity_factor == pytest.approx(1.0)
