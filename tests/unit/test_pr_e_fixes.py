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

    from legoesm.atmosphere.dynamics import rce_surface_flux as m

    assert m._RCEMIP1_C_H == 1.5e-3
    assert m._RCEMIP1_GUSTINESS_FLOOR_MS == 5.0
    for fn in (m.compose_rce_surface_scalar_tendencies, m.apply_rce_surface_fluxes):
        sig = inspect.signature(fn)
        assert sig.parameters["C_h"].default == m._RCEMIP1_C_H
        assert sig.parameters["gustiness_floor"].default == m._RCEMIP1_GUSTINESS_FLOOR_MS
