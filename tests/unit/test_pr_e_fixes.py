"""PR E: run_coupled --microphysics choices hardening + RCE closure constants."""
from __future__ import annotations

import sys

import pytest


def test_run_coupled_rejects_unknown_microphysics(monkeypatch):
    """``--microphysics`` now carries argparse ``choices=`` (like every sibling
    scheme flag), so a typo fails at PARSE time instead of only surfacing later
    inside ExperimentConfig.validate_strict at driver setup."""
    from scripts.run.run_coupled import main

    monkeypatch.setattr(
        sys, "argv",
        ["run_coupled.py", "--preset", "aquaplanet", "--microphysics", "morrson"],
    )
    with pytest.raises(SystemExit):
        main()


def test_rce_surface_flux_defaults_reference_named_constants():
    """The RCEMIP1 closure coefficients are module constants (not inline
    signature literals); the composer defaults bind to them."""
    import inspect

    from legoesm.atmosphere.dynamics.crm import rce_surface_flux as m

    assert m._RCEMIP1_C_H == 1.5e-3
    assert m._RCEMIP1_GUSTINESS_FLOOR_MS == 5.0
    for fn in (m.compose_rce_surface_scalar_tendencies, m.apply_rce_surface_fluxes):
        sig = inspect.signature(fn)
        assert sig.parameters["C_h"].default == m._RCEMIP1_C_H
        assert sig.parameters["gustiness_floor"].default == m._RCEMIP1_GUSTINESS_FLOOR_MS


def test_cli_microphysics_choices_match_validate_strict():
    """The run_coupled --microphysics choices derive from VALID_MICROPHYSICS, the
    SAME set ExperimentConfig.validate_strict enforces, so the CLI cannot drop an
    advertised scheme (codex caught ml_emulator missing) or drift from the config.
    """
    from legoesm.driver.config import VALID_MICROPHYSICS, ExperimentConfig

    # The scheme codex caught the hardcoded list dropping:
    assert "ml_emulator" in VALID_MICROPHYSICS

    # validate_strict accepts every scheme in the set (it raises a single
    # ValueError listing all problems; assert microphysics is never among them).
    for scheme in VALID_MICROPHYSICS:
        try:
            ExperimentConfig(microphysics=scheme).validate_strict()
        except ValueError as e:
            assert "microphysics must be one of" not in str(e), (
                f"{scheme!r} is in VALID_MICROPHYSICS but validate_strict "
                f"rejects it: {e}"
            )

    # ... and an unknown scheme is rejected by validate_strict.
    with pytest.raises(ValueError, match="microphysics must be one of"):
        ExperimentConfig(microphysics="bogus_micro").validate_strict()
